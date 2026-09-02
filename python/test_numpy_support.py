"""Cross-language checks for PiP 4.0 alpha array documents."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from .numpy_support import to_ndarray
except ModuleNotFoundError:
    np = None
    to_ndarray = None


@unittest.skipUnless(np is not None, "NumPy is not installed")
class NumpySupportTests(unittest.TestCase):
    def decode(self, payload) -> np.ndarray:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            path.write_text(json.dumps(payload))
            return to_ndarray(path)

    def test_dense_universal_tensor(self) -> None:
        array = self.decode({
            "storage": "dense",
            "tensor": {
                "kind": "tensor", "version": 1, "scalar": "i64",
                "shape": [2, 2], "data": [1, 2, 3, 4],
            },
        })
        np.testing.assert_array_equal(array, np.array([[1, 2], [3, 4]]))

    def test_sparse_universal_tensor(self) -> None:
        array = self.decode({
            "storage": "sparse",
            "tensor": {
                "kind": "tensor_sparse", "version": 1, "scalar": "f64",
                "shape": [2, 3], "indices": [1, 5], "values": [7.0, -4.0],
            },
        })
        np.testing.assert_array_equal(array, np.array([[0.0, 7.0, 0.0], [0.0, 0.0, -4.0]]))

    def test_sparse_indices_must_be_canonical(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.decode({
                "storage": "sparse",
                "tensor": {
                    "kind": "tensor_sparse", "version": 1, "scalar": "i64",
                    "shape": [4], "indices": [2, 1], "values": [3, 4],
                },
            })

    def test_square_lattice_uses_geometry_shape(self) -> None:
        array = self.decode({
            "geometry": {"shape": [2, 2], "boundary": "periodic", "spacing": [1.0, 1.0]},
            "values": [1, 2, 3, 4],
            "initialization_rng": None,
        })
        np.testing.assert_array_equal(array, np.array([[1, 2], [3, 4]]))

    def test_pre_4_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported PiP 4.0 alpha"):
            self.decode({"kind": "tensor", "shape": [1], "data": [1]})


if __name__ == "__main__":
    unittest.main()

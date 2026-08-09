"""Cross-language checks for the current versioned PiP JSON schemas."""

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

    def test_square_lattice_tags_decode_to_ranked_arrays(self) -> None:
        for kind in ("square_lattice_periodic", "square_lattice_reflective"):
            array = self.decode(
                {
                    "kind": kind,
                    "version": 1,
                    "scalar": "i64",
                    "shape": [2, 2],
                    "data": [1, 2, 3, 4],
                }
            )
            np.testing.assert_array_equal(array, np.array([[1, 2], [3, 4]]))

    def test_sparse_tensor_and_matrix_decode_without_dense_json_data(self) -> None:
        for kind in ("tensor_sparse", "matrix_sparse"):
            array = self.decode(
                {
                    "kind": kind,
                    "version": 1,
                    "scalar": "i64",
                    "shape": [2, 3],
                    "indices": [1, 5],
                    "values": [7, -4],
                }
            )
            np.testing.assert_array_equal(array, np.array([[0, 7, 0], [0, 0, -4]]))

    def test_sparse_indices_must_be_canonical(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.decode(
                {
                    "kind": "tensor_sparse",
                    "version": 1,
                    "scalar": "i64",
                    "shape": [4],
                    "indices": [2, 1],
                    "values": [3, 4],
                }
            )

    def test_complex_scalar_metadata_restores_numeric_dtype(self) -> None:
        array = self.decode(
            {
                "kind": "tensor",
                "version": 1,
                "scalar": "complex_f64",
                "shape": [2],
                "data": [[1.5, -2.0], [0.0, 3.0]],
            }
        )
        self.assertEqual(array.dtype, np.dtype(np.complex128))
        np.testing.assert_array_equal(array, np.array([1.5 - 2.0j, 3.0j]))

    def test_numeric_metadata_rejects_lossy_or_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid PiP i64 scalar"):
            self.decode(
                {
                    "kind": "tensor",
                    "version": 1,
                    "scalar": "i64",
                    "shape": [1],
                    "data": [1.5],
                }
            )
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            self.decode(
                {
                    "kind": "tensor",
                    "version": 1,
                    "scalar": "f64",
                    "shape": [1],
                    "data": [float("inf")],
                }
            )


if __name__ == "__main__":
    unittest.main()

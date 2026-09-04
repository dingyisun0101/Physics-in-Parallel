"""Cross-language checks for PiP 4.0 alpha array documents."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from .numpy_support import to_ndarray, payload_to_ndarray
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

    def test_rust_generated_fixtures(self) -> None:
        documents = json.loads(Path(__file__).with_name("fixtures.json").read_text())
        arrays = [payload_to_ndarray(document) for document in documents]
        np.testing.assert_array_equal(arrays[0], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(arrays[1], [[0, 2], [0, -4]])
        self.assertEqual(sorted(arrays[2].flat), [1, 2, 3, 4])
        for document in documents:
            malformed = json.loads(json.dumps(document))
            if "tensor" in malformed:
                key = "data" if malformed["backend"] == "dense" else "values"
                malformed["tensor"][key][0] = None
            else:
                malformed["values"][0] = None
            with self.assertRaises(ValueError):
                payload_to_ndarray(malformed)

    def test_dense_universal_tensor(self) -> None:
        array = self.decode({
            "backend": "dense",
            "tensor": {
                "kind": "tensor", "version": 2, "scalar": "i64",
                "shape": [2, 2], "data": [1, 2, 3, 4],
            },
        })
        np.testing.assert_array_equal(array, np.array([[1, 2], [3, 4]]))

    def test_sparse_universal_tensor(self) -> None:
        array = self.decode({
            "backend": "sparse",
            "tensor": {
                "kind": "tensor_sparse", "version": 2, "scalar": "f64",
                "shape": [2, 3], "indices": [1, 5], "values": [7.0, -4.0],
            },
        })
        np.testing.assert_array_equal(array, np.array([[0.0, 7.0, 0.0], [0.0, 0.0, -4.0]]))

    def test_sparse_indices_must_be_canonical(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.decode({
                "backend": "sparse",
                "tensor": {
                    "kind": "tensor_sparse", "version": 2, "scalar": "i64",
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

    def test_alpha_1_tensor_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 2, got 1"):
            self.decode({
                "backend": "dense",
                "tensor": {
                    "kind": "tensor", "version": 1, "scalar": "i64",
                    "shape": [1], "data": [1],
                },
            })

        with self.assertRaisesRegex(ValueError, "unsupported PiP 4.0 alpha"):
            self.decode({
                "storage": "dense",
                "tensor": {
                    "kind": "tensor", "version": 1, "scalar": "i64",
                    "shape": [1], "data": [1],
                },
            })

    def test_checked_wide_integers_and_complex_narrowing(self) -> None:
        def document(scalar, value):
            return {"backend": "dense", "tensor": {"kind": "tensor", "version": 2,
                    "scalar": scalar, "shape": [1], "data": [value]}}
        for scalar, low, high in [("i128", -(1 << 127), (1 << 127) - 1),
                                   ("u128", 0, (1 << 128) - 1)]:
            for value in [low, high]:
                self.assertEqual(payload_to_ndarray(document(scalar, value))[0], value)
            for value in [low - 1, high + 1, True, 1.5, "3", None]:
                with self.assertRaises(ValueError):
                    payload_to_ndarray(document(scalar, value))
        with self.assertRaisesRegex(ValueError, "overflow"):
            payload_to_ndarray(document("complex_f32", [1e100, 0]))
        self.assertEqual(payload_to_ndarray(document("complex_f32", [1, -2]))[0], 1 - 2j)

    def test_lattice_validation_and_complex_inference(self) -> None:
        def document(value):
            return {"geometry": {"shape": [1]}, "values": [value], "initialization_rng": None}
        for value in [True, None, "1", float("inf"), [1, float("nan")], [1, 2, 3]]:
            with self.assertRaises(ValueError):
                payload_to_ndarray(document(value))
        self.assertEqual(payload_to_ndarray(document([1, 2]))[0], 1 + 2j)
        self.assertEqual(payload_to_ndarray(document((1 << 128) - 1))[0], (1 << 128) - 1)

    def test_sparse_expansion_limit_precedes_allocation(self) -> None:
        document = {"backend": "sparse", "tensor": {"kind": "tensor_sparse", "version": 2,
                    "scalar": "f64", "shape": [10**12], "indices": [], "values": []}}
        with self.assertRaisesRegex(ValueError, "max_elements"):
            payload_to_ndarray(document, max_elements=100)
        for cap in [-1, True, 2.5]:
            with self.assertRaises(ValueError):
                payload_to_ndarray(document, max_elements=cap)


if __name__ == "__main__":
    unittest.main()

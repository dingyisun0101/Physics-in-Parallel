"""NumPy helpers for reading PiP JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_FLAT_PAYLOAD_KINDS = {
    "tensor",
    "tensor_sparse",
    "tensor_2d",
    "matrix",
    "vector_list",
    "grid",
    "grid_periodic",
    "grid_clamped",
}


def to_ndarray(path) -> np.ndarray:
    """
    Read one PiP JSON output file into a NumPy array.

    Behavior:
    - Flat PiP payloads (`kind` + `shape` + `data`) are converted to numeric
      ndarrays using the provided shape.
    - Legacy PiP payloads are supported for backward compatibility.
    - Composite PiP payloads such as `PhysObj` are returned as 0-D object
      arrays containing recursively converted Python/NumPy content.
    """
    payload = json.loads(Path(path).read_text())
    return _payload_to_ndarray(payload)


def _payload_to_ndarray(payload) -> np.ndarray:
    if isinstance(payload, dict):
        kind = payload.get("kind")

        # Current PiP schema
        if kind in _FLAT_PAYLOAD_KINDS and {"shape", "data"}.issubset(payload.keys()):
            return _flat_payload_to_ndarray(payload)

        # Legacy schema compatibility
        if kind == "tensor" and "storage" in payload:
            return _legacy_tensor_payload_to_ndarray(payload)
        if kind == "vector_list" and "storage" in payload:
            return _legacy_vector_list_payload_to_ndarray(payload)
        if kind == "grid" and "storage" in payload:
            return _legacy_grid_payload_to_ndarray(payload)
        if _looks_like_legacy_compact_grid(payload):
            return _legacy_compact_grid_payload_to_ndarray(payload)

        return _object_scalar_array(_json_to_python(payload))

    if isinstance(payload, list):
        return np.asarray([_json_to_python(item) for item in payload], dtype=object)

    return np.asarray(payload)


def _flat_payload_to_ndarray(payload) -> np.ndarray:
    _require_keys(payload, {"shape", "data"})
    shape = _normalize_shape(payload["shape"])

    array = np.asarray(payload["data"])
    expected_size = int(np.prod(shape, dtype=np.int64))
    if array.size != expected_size:
        raise ValueError(
            f"flat payload data length mismatch: expected {expected_size}, got {array.size}"
        )

    return array.reshape(shape)


def _legacy_tensor_payload_to_ndarray(payload) -> np.ndarray:
    _require_keys(payload, {"shape", "storage", "data"})
    shape = _normalize_shape(payload["shape"])
    storage = payload["storage"]

    if storage == "dense":
        array = np.asarray(payload["data"])
        return array.reshape(shape)

    if storage == "sparse":
        entries = payload["data"]["entries"]
        dtype = _numpy_dtype_from_scalar_type(payload.get("scalar_type"))
        array = np.zeros(shape, dtype=dtype)
        for entry in entries:
            array.flat[entry["index"]] = entry["value"]
        return array

    raise ValueError(f"unsupported legacy PiP tensor storage: {storage!r}")


def _legacy_vector_list_payload_to_ndarray(payload) -> np.ndarray:
    _require_keys(payload, {"shape", "data"})
    dim, n = _normalize_shape(payload["shape"])
    array = np.asarray(payload["data"])
    if array.shape != (n, dim):
        raise ValueError(
            f"legacy vector_list payload data shape mismatch: expected {(n, dim)}, got {array.shape}"
        )
    # Legacy convention was logical [dim, n].
    return array.T


def _legacy_grid_payload_to_ndarray(payload) -> np.ndarray:
    _require_keys(payload, {"shape", "data"})
    return _reshape_legacy_grid_data(payload["shape"], payload["data"])


def _legacy_compact_grid_payload_to_ndarray(payload) -> np.ndarray:
    return _reshape_legacy_grid_data(payload["shape"], payload["data"])


def _reshape_legacy_grid_data(shape_metadata, data) -> np.ndarray:
    d, l = _normalize_shape(shape_metadata)
    array = np.asarray(data)
    return array.reshape((l,) * d)


def _looks_like_legacy_compact_grid(payload) -> bool:
    if set(payload.keys()) != {"shape", "data"}:
        return False
    shape = payload["shape"]
    return (
        isinstance(shape, list)
        and len(shape) == 2
        and all(isinstance(value, int) for value in shape)
    )


def _normalize_shape(shape) -> tuple[int, ...]:
    if not isinstance(shape, list) or not shape or not all(isinstance(dim, int) for dim in shape):
        raise ValueError(f"invalid PiP shape metadata: {shape!r}")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"PiP shape dimensions must be > 0: {shape!r}")
    return tuple(shape)


def _require_keys(payload, required_keys) -> None:
    missing = required_keys.difference(payload.keys())
    if missing:
        raise ValueError(f"missing required PiP payload keys: {sorted(missing)}")


def _numpy_dtype_from_scalar_type(scalar_type):
    if scalar_type in {"f32"}:
        return np.float32
    if scalar_type in {"f64"}:
        return np.float64
    if scalar_type in {"i8"}:
        return np.int8
    if scalar_type in {"i16"}:
        return np.int16
    if scalar_type in {"i32"}:
        return np.int32
    if scalar_type in {"i64", "isize"}:
        return np.int64
    if scalar_type in {"u8"}:
        return np.uint8
    if scalar_type in {"u16"}:
        return np.uint16
    if scalar_type in {"u32"}:
        return np.uint32
    if scalar_type in {"u64", "usize"}:
        return np.uint64
    if isinstance(scalar_type, str) and "Complex<f32>" in scalar_type:
        return np.complex64
    if isinstance(scalar_type, str) and "Complex<f64>" in scalar_type:
        return np.complex128
    return np.float64


def _json_to_python(value):
    if isinstance(value, dict):
        return {key: _json_to_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_to_python(item) for item in value]
    return value


def _object_scalar_array(value) -> np.ndarray:
    array = np.empty((), dtype=object)
    array[()] = value
    return array

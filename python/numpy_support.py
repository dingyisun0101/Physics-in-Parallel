"""NumPy helpers for reading PiP JSON outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


_FLAT_PAYLOAD_KINDS = {
    "tensor",
    "tensor_2d",
    "matrix",
    "vector_list",
    "grid",
    "grid_periodic",
    "grid_clamped",
    "square_lattice_periodic",
    "square_lattice_reflective",
}

_SPARSE_PAYLOAD_KINDS = {"tensor_sparse", "matrix_sparse"}
_JSON_SCHEMA_VERSION = 1


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
        if kind in _SPARSE_PAYLOAD_KINDS:
            return _sparse_payload_to_ndarray(payload)
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
    _require_keys(payload, {"version", "scalar", "shape", "data"})
    _require_current_version(payload)
    shape = _normalize_shape(payload["shape"])

    array = _numeric_values(payload["data"], payload["scalar"])
    expected_size = math.prod(shape)
    if array.size != expected_size:
        raise ValueError(
            f"flat payload data length mismatch: expected {expected_size}, got {array.size}"
        )

    return array.reshape(shape)


def _sparse_payload_to_ndarray(payload) -> np.ndarray:
    _require_keys(payload, {"version", "scalar", "shape", "indices", "values"})
    _require_current_version(payload)
    shape = _normalize_shape(payload["shape"])
    indices = payload["indices"]
    values = payload["values"]
    if not isinstance(indices, list) or not isinstance(values, list):
        raise ValueError("sparse PiP indices and values must be arrays")
    if len(indices) != len(values):
        raise ValueError(
            f"sparse PiP entry length mismatch: {len(indices)} indices, {len(values)} values"
        )
    logical_size = math.prod(shape)
    previous = -1
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"invalid sparse PiP index: {index!r}")
        if index <= previous:
            raise ValueError("sparse PiP indices must be strictly increasing")
        if index >= logical_size:
            raise ValueError(
                f"sparse PiP index out of bounds: {index} >= {logical_size}"
            )
        previous = index
    numeric_values = _numeric_values(values, payload["scalar"])
    if np.any(numeric_values == 0):
        raise ValueError("sparse PiP values must not contain explicit zeros")
    array = np.zeros(logical_size, dtype=numeric_values.dtype)
    if indices:
        array[np.asarray(indices, dtype=np.intp)] = numeric_values
    return array.reshape(shape)


def _numeric_values(values, scalar) -> np.ndarray:
    if not isinstance(values, list):
        raise ValueError("PiP numeric payload values must be an array")
    if scalar in {"complex_f32", "complex_f64"}:
        converted = []
        for value in values:
            if isinstance(value, dict) and set(value) == {"re", "im"}:
                real, imaginary = value["re"], value["im"]
            elif isinstance(value, list) and len(value) == 2:
                real, imaginary = value
            else:
                raise ValueError(f"invalid PiP complex scalar: {value!r}")
            if not _is_finite_number(real) or not _is_finite_number(imaginary):
                raise ValueError(f"invalid PiP complex scalar: {value!r}")
            converted.append(complex(real, imaginary))
        dtype = np.complex64 if scalar == "complex_f32" else np.complex128
        array = np.asarray(converted, dtype=dtype)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"PiP {scalar} values overflow their declared dtype")
        return array

    integer_dtypes = {
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "isize": np.intp,
        "u8": np.uint8,
        "u16": np.uint16,
        "u32": np.uint32,
        "u64": np.uint64,
        "usize": np.uintp,
    }
    if scalar in integer_dtypes:
        dtype = integer_dtypes[scalar]
        _validate_integer_values(values, scalar, np.iinfo(dtype).min, np.iinfo(dtype).max)
        return np.asarray(values, dtype=dtype)
    if scalar in {"i128", "u128"}:
        minimum = -(1 << 127) if scalar == "i128" else 0
        maximum = (1 << 127) - 1 if scalar == "i128" else (1 << 128) - 1
        _validate_integer_values(values, scalar, minimum, maximum)
        return np.asarray(values, dtype=object)

    float_dtypes = {
        "f32": np.float32,
        "f64": np.float64,
    }
    try:
        dtype = float_dtypes[scalar]
    except KeyError as error:
        raise ValueError(f"unsupported PiP scalar kind: {scalar!r}") from error
    if not all(_is_finite_number(value) for value in values):
        raise ValueError(f"PiP {scalar} values must contain only finite numbers")
    array = np.asarray(values, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"PiP {scalar} values overflow their declared dtype")
    return array


def _validate_integer_values(values, scalar, minimum, maximum) -> None:
    for value in values:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
            or value > maximum
        ):
            raise ValueError(f"invalid PiP {scalar} scalar: {value!r}")


def _is_finite_number(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


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
    if (
        not isinstance(shape, list)
        or not shape
        or not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape)
    ):
        raise ValueError(f"invalid PiP shape metadata: {shape!r}")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"PiP shape dimensions must be > 0: {shape!r}")
    return tuple(shape)


def _require_keys(payload, required_keys) -> None:
    missing = required_keys.difference(payload.keys())
    if missing:
        raise ValueError(f"missing required PiP payload keys: {sorted(missing)}")


def _require_current_version(payload) -> None:
    version = payload.get("version")
    if version != _JSON_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported PiP JSON schema version: expected {_JSON_SCHEMA_VERSION}, got {version!r}"
        )


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

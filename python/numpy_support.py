"""NumPy reader for direct PiP 4.1 alpha JSON documents."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


_JSON_SCHEMA_VERSION = 2


def to_ndarray(path, *, max_elements=None) -> np.ndarray:
    """Read a PiP JSON file; optionally cap the dense result's element count.

    The cap is checked before NumPy allocation, including sparse expansion.
    JSON parsing itself still materializes the file. See payload_to_ndarray
    for decoded-document validation and lattice dtype inference.
    """
    payload = json.loads(Path(path).read_text())
    return payload_to_ndarray(payload, max_elements=max_elements)


def payload_to_ndarray(payload, *, max_elements=None) -> np.ndarray:
    """Validate a decoded PiP document and return a dense NumPy array.

    Tensor dtypes follow the scalar tag, with checked i128/u128 object arrays.
    Lattices have no scalar tag: real data uses NumPy numeric inference and
    [real, imaginary] pairs become complex128. Original scalar width cannot be
    recovered from a lattice document. Booleans and nonfinite data are rejected.
    max_elements must be a nonnegative integer or None; it bounds element count,
    not JSON parsing memory or object-array payload bytes. Errors are ValueError.
    """
    if max_elements is not None and (not isinstance(max_elements, int)
            or isinstance(max_elements, bool) or max_elements < 0):
        raise ValueError("max_elements must be a nonnegative integer or None")
    if not isinstance(payload, dict):
        raise ValueError("PiP array document must be a JSON object")

    if set(payload) == {"backend", "tensor"}:
        return _universal_tensor_to_ndarray(payload, max_elements)
    if {"geometry", "values", "initialization_rng"}.issubset(payload):
        return _lattice_to_ndarray(payload, max_elements)
    raise ValueError("unsupported PiP 4.1 alpha array document")


def _universal_tensor_to_ndarray(payload, max_elements) -> np.ndarray:
    backend = payload["backend"]
    tensor = payload["tensor"]
    if not isinstance(tensor, dict):
        raise ValueError("PiP tensor payload must be a JSON object")
    if backend == "dense":
        return _dense_tensor_to_ndarray(tensor, max_elements)
    if backend == "sparse":
        return _sparse_tensor_to_ndarray(tensor, max_elements)
    raise ValueError(f"unsupported PiP tensor backend: {backend!r}")


def _dense_tensor_to_ndarray(payload, max_elements) -> np.ndarray:
    _require_exact_keys(payload, {"kind", "version", "scalar", "shape", "data"})
    if payload["kind"] != "tensor":
        raise ValueError(f"dense PiP tensor kind must be 'tensor', got {payload['kind']!r}")
    _require_current_version(payload)
    shape = _normalize_shape(payload["shape"], max_elements)
    array = _numeric_values(payload["data"], payload["scalar"])
    expected = math.prod(shape)
    if array.size != expected:
        raise ValueError(f"tensor data length mismatch: expected {expected}, got {array.size}")
    return array.reshape(shape)


def _sparse_tensor_to_ndarray(payload, max_elements) -> np.ndarray:
    required = {"kind", "version", "scalar", "shape", "indices", "values"}
    _require_exact_keys(payload, required)
    if payload["kind"] != "tensor_sparse":
        raise ValueError(
            f"sparse PiP tensor kind must be 'tensor_sparse', got {payload['kind']!r}"
        )
    _require_current_version(payload)
    shape = _normalize_shape(payload["shape"], max_elements)
    indices = payload["indices"]
    values = payload["values"]
    if not isinstance(indices, list) or not isinstance(values, list):
        raise ValueError("sparse PiP indices and values must be arrays")
    if len(indices) != len(values):
        raise ValueError("sparse PiP indices and values must have equal length")

    logical_size = math.prod(shape)
    previous = -1
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"invalid sparse PiP index: {index!r}")
        if index <= previous:
            raise ValueError("sparse PiP indices must be strictly increasing")
        if index >= logical_size:
            raise ValueError(f"sparse PiP index out of bounds: {index} >= {logical_size}")
        previous = index

    numeric = _numeric_values(values, payload["scalar"])
    if np.any(numeric == 0):
        raise ValueError("sparse PiP values must not contain explicit zeros")
    array = np.zeros(logical_size, dtype=numeric.dtype)
    if indices:
        array[np.asarray(indices, dtype=np.intp)] = numeric
    return array.reshape(shape)


def _lattice_to_ndarray(payload, max_elements) -> np.ndarray:
    _require_exact_keys(payload, {"geometry", "values", "initialization_rng"})
    geometry = payload["geometry"]
    if not isinstance(geometry, dict) or "shape" not in geometry:
        raise ValueError("PiP lattice geometry must contain shape")
    shape = _normalize_shape(geometry["shape"], max_elements)
    values = payload["values"]
    if not isinstance(values, list) or len(values) != math.prod(shape):
        raise ValueError("PiP lattice value count does not match geometry")
    if any(isinstance(value, list) for value in values):
        return _numeric_values(values, "complex_f64").reshape(shape)
    if not all(_is_finite_number(value) for value in values):
        raise ValueError("PiP lattice values must contain only finite numbers")
    if all(isinstance(value, int) for value in values):
        if any(value < -(1 << 127) or value >= (1 << 128) for value in values):
            raise ValueError("PiP lattice integer is outside supported 128-bit ranges")
        # Avoid lossy float inference for mixtures of large signed/unsigned ints.
        low, high = min(values), max(values)
        dtype = np.int64 if -(1 << 63) <= low and high < (1 << 63) else (
            np.uint64 if 0 <= low and high < (1 << 64) else object)
        return np.asarray(values, dtype=dtype).reshape(shape)
    return _numeric_values(values, "f64").reshape(shape)


def _numeric_values(values, scalar) -> np.ndarray:
    if not isinstance(values, list):
        raise ValueError("PiP numeric values must be an array")
    if scalar in {"complex_f32", "complex_f64"}:
        converted = []
        for value in values:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"invalid PiP complex scalar: {value!r}")
            if not all(_is_finite_number(component) for component in value):
                raise ValueError(f"invalid PiP complex scalar: {value!r}")
            converted.append(complex(value[0], value[1]))
        dtype = np.complex64 if scalar == "complex_f32" else np.complex128
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(converted, dtype=dtype)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"PiP {scalar} values overflow their declared dtype")
        return array

    integer_dtypes = {
        "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64,
        "isize": np.intp, "u8": np.uint8, "u16": np.uint16,
        "u32": np.uint32, "u64": np.uint64, "usize": np.uintp,
    }
    if scalar in integer_dtypes:
        dtype = integer_dtypes[scalar]
        limits = np.iinfo(dtype)
        for value in values:
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < limits.min or value > limits.max):
                raise ValueError(f"invalid PiP {scalar} scalar: {value!r}")
        return np.asarray(values, dtype=dtype)
    if scalar in {"i128", "u128"}:
        low, high = (-(1 << 127), (1 << 127) - 1) if scalar == "i128" else (0, (1 << 128) - 1)
        if any(not isinstance(value, int) or isinstance(value, bool)
               or not low <= value <= high for value in values):
            raise ValueError(f"invalid PiP {scalar} scalar")
        return np.asarray(values, dtype=object)

    float_dtypes = {"f32": np.float32, "f64": np.float64}
    if scalar not in float_dtypes:
        raise ValueError(f"unsupported PiP scalar kind: {scalar!r}")
    if not all(_is_finite_number(value) for value in values):
        raise ValueError(f"PiP {scalar} values must contain only finite numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(values, dtype=float_dtypes[scalar])
    if not np.all(np.isfinite(array)):
        raise ValueError(f"PiP {scalar} values overflow their declared dtype")
    return array


def _normalize_shape(shape, max_elements) -> tuple[int, ...]:
    if (not isinstance(shape, list) or not shape
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in shape)
            or any(value <= 0 for value in shape)):
        raise ValueError(f"invalid PiP shape metadata: {shape!r}")
    size = math.prod(shape)
    if size > np.iinfo(np.intp).max:
        raise ValueError("PiP shape exceeds NumPy signed index capacity")
    if max_elements is not None and size > max_elements:
        raise ValueError(f"PiP dense result exceeds max_elements={max_elements}")
    return tuple(shape)


def _require_exact_keys(payload, expected) -> None:
    if set(payload) != expected:
        raise ValueError(f"invalid PiP payload fields: expected {sorted(expected)}")


def _require_current_version(payload) -> None:
    if payload["version"] != _JSON_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported PiP JSON schema version: expected {_JSON_SCHEMA_VERSION}, "
            f"got {payload['version']!r}"
        )


def _is_finite_number(value) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False

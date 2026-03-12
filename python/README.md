# PiP Python Helpers

`numpy_support.py` provides one public loader:

```python
from numpy_support import to_ndarray
```

## Public API

```python
to_ndarray(path) -> numpy.ndarray
```

`path` must point to a JSON file produced by PiP.

## Current PiP Flat Schema

PiP now serializes numeric payloads as:

```json
{
  "kind": "...",
  "shape": [...],
  "data": [...]
}
```

`to_ndarray(path)` reshapes `data` using `shape` (NumPy convention, row-major/C order).

## Supported Kinds

### Dense tensor

```json
{
  "kind": "tensor",
  "shape": [2, 2],
  "data": [1.0, 2.0, 3.0, 4.0]
}
```

Output:

```python
arr.shape == (2, 2)
```

### Sparse tensor (serialized densely)

```json
{
  "kind": "tensor_sparse",
  "shape": [2, 3],
  "data": [0.0, 2.0, 0.0, 0.0, 0.0, 5.0]
}
```

Output:

```python
arr.shape == (2, 3)
```

### Tensor2D / Matrix

```json
{
  "kind": "matrix",
  "shape": [2, 3],
  "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
}
```

Output:

```python
arr.shape == (2, 3)
```

### VectorList

Current VectorList convention is `[n, dim]`.

```json
{
  "kind": "vector_list",
  "shape": [2, 3],
  "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
}
```

Output:

```python
arr.shape == (2, 3)
```

### Grid

`kind` indicates boundary mode:
- `grid_periodic`
- `grid_clamped`

Shape follows ndarray axes directly (e.g. 2D side length 4 is `[4, 4]`).

```json
{
  "kind": "grid_periodic",
  "shape": [4, 4],
  "data": [...]
}
```

Output:

```python
arr.shape == (4, 4)
```

## Composite Outputs

Some PiP JSON outputs, such as `PhysObj`, are not one homogeneous numeric array.
For those, `to_ndarray(path)` returns a 0-D object array containing recursively
converted Python data.

Example:

```python
arr = to_ndarray("phys_obj.json")
payload = arr[()]
```

## Legacy Compatibility

`to_ndarray()` still accepts older PiP payloads with fields like `storage`,
`sparse.entries`, and compact grid metadata (`shape = [d, l]`).

## Requirements

`numpy_support.py` requires `numpy` to be installed in the Python environment.

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

## Supported PiP JSON Kinds

### Dense tensor

Input shape:

```json
{
  "kind": "tensor",
  "shape": [2, 2],
  "storage": "dense",
  "data": [1.0, 2.0, 3.0, 4.0]
}
```

Output:

```python
arr.shape == (2, 2)
```

### Sparse tensor

Input shape:

```json
{
  "kind": "tensor",
  "shape": [2, 3],
  "storage": "sparse",
  "data": {
    "nnz": 2,
    "entries": [
      {"index": 1, "value": 2.0},
      {"index": 5, "value": 5.0}
    ]
  }
}
```

Output:

```python
arr.shape == (2, 3)
```

### Tensor2D / Matrix

Input shape:

```json
{
  "kind": "matrix",
  "shape": [2, 3],
  "storage": "dense",
  "data": [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
  ]
}
```

Output:

```python
arr.shape == (2, 3)
```

### VectorList

PiP stores vector-list data as `n` row vectors, but its logical ndarray convention
is `[dim, n]`. `to_ndarray()` returns that logical shape.

Input shape:

```json
{
  "kind": "vector_list",
  "shape": [3, 2],
  "storage": "dense",
  "data": [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
  ]
}
```

Output:

```python
arr.shape == (3, 2)
```

### Grid

Input shape:

```json
{
  "kind": "grid",
  "shape": [2, 4],
  "storage": "dense",
  "data": [...]
}
```

Interpretation:
- `shape[0] = d`
- `shape[1] = l`
- returned ndarray shape is `(l,) * d`

So `[2, 4]` becomes:

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

## Requirements

`numpy_support.py` requires `numpy` to be installed in the Python environment.

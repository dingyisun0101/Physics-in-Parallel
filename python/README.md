# PiP NumPy Helper

`numpy_support.py` reads JSON produced by direct PiP 4.0 Serde:

```python
from numpy_support import to_ndarray

array = to_ndarray("tensor.json")
```

It supports universal `Tensor`, `Matrix`, and `VectorList` documents in both
dense and sparse representations, plus `SquareLattice` documents. Sparse
values are materialized as a NumPy array at this external analysis boundary.

The helper intentionally has no pre-4.0 compatibility paths. It validates
schema version, scalar type, shape, canonical sparse indices, and finite
numeric values.

Run the tests with:

```bash
python -m unittest python/test_numpy_support.py
```

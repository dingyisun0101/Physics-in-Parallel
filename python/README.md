# PiP NumPy Helper

> **Alpha schema:** this helper targets PiP `4.0.0-alpha.1`. It does not read
> PiP 3.x documents, and its accepted schema may change between alpha releases.

`numpy_support.py` reads JSON produced by direct PiP 4.0 alpha Serde:

```python
from numpy_support import to_ndarray

array = to_ndarray("tensor.json")
```

It supports universal `Tensor`, `Matrix`, and `VectorList` documents in both
dense and sparse representations, plus `SquareLattice` documents. Sparse
values are materialized as a NumPy array at this external analysis boundary.

The helper validates schema version, scalar type, shape, canonical sparse
indices, and finite numeric values. It is an external analysis adapter, not a
second PiP serialization contract.

Run the tests with:

```bash
python -m unittest python/test_numpy_support.py
```

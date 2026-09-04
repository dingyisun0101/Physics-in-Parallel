# PiP NumPy Helper

> **Alpha schema:** this helper targets PiP `4.0.0-alpha.2`. It does not read
> PiP 3.x documents, and its accepted schema may change between alpha releases.

`numpy_support.py` reads JSON produced by direct PiP 4.0 alpha Serde:

```python
from numpy_support import to_ndarray

array = to_ndarray("tensor.json")
```

It supports universal `Tensor`, `Matrix`, and `VectorList` documents in both
dense and sparse backends, plus `SquareLattice` documents. Sparse values are
materialized as a NumPy array at this external analysis boundary. Tensor-family
documents use schema version 2 and an explicit `backend` discriminator.

The helper validates schema version, scalar type, shape, canonical sparse
indices, and finite numeric values. It is an external analysis adapter, not a
second PiP serialization contract.

Run the tests with:

```bash
python -m unittest python/test_numpy_support.py
```

For already-decoded JSON, call `payload_to_ndarray(payload)`. Both entry points
accept `max_elements=...`, a nonnegative dense-result element limit checked
before NumPy allocation. This includes sparse expansion; it does not bound JSON
parsing memory or Python object sizes.

The helper range-checks signed/unsigned 128-bit integers before creating object
arrays and checks complex finiteness after narrowing. Lattices lack scalar type
metadata: integer values infer int64/uint64/object as needed, mixed real values
become float64, and complex pairs become complex128. Original scalar width
cannot be recovered. Booleans, malformed numbers and nonfinite values fail.

`fixtures.json` contains Rust-produced documents tested alongside malformed
variants. Regenerate with `cargo run --example numpy_fixtures > python/fixtures.json`.
These checks cover supported cases, rather than claiming complete Serde parity.

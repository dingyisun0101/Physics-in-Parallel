# PiP Advanced API

The advanced API is opt-in:

```rust
use physics_in_parallel::prelude::advanced::*;
```

Use a domain's basic API first. Use this layer only for missing coverage or a
measured efficiency requirement. PiP's crate-private implementation modules
remain inaccessible.

## Exported Symbols

Raw container storage:

- `RawStorage`

Structured matrix representations:

- `DiagonalMatrix`
- `SymmetricMatrix`
- `AntiSymmetricMatrix`
- `UpperTriangularMatrix`
- `StrictUpperTriangularMatrix`
- `LowerTriangularMatrix`
- `StrictLowerTriangularMatrix`

Generic object and interaction engines:

- `AttrId`, `AttrsMeta`, `AttrsCore`, `AttrsError`, `PhysObjAdvanced`
- `ObjId`, `InteractionId`, `InteractionNodes`, `InteractionOrder`
- `InteractionTopology`, `Interaction`, `InteractionError`
- `NeighborList`, `NeighborListError`

Sampling and square-lattice extensions:

- `DynamicWeightedIndex`, `DynamicWeightedIndexError`
- `Kernel`, `NearestNeighborKernel`, `PowerLawKernel`
- `UniformDistanceKernel`, `SquareLatticeAdvanced`

## Raw Storage

`RawStorage<T>` provides flat-index reads and writes, optional dense slices,
and stored-entry extraction for `Tensor<T>`, `Matrix<T>`, and `VectorList<T>`.
The ordinary coordinate-based methods should be considered first.

Representation matters here:

- dense slices exist only for dense storage
- stored-entry traversal is `O(total elements)` for dense storage
- stored-entry traversal is `O(stored elements)` for sparse storage
- flat writes to sparse storage may insert or remove entries

Code using this trait deliberately accepts backend-sensitive costs.

## Generic Engines

`AttrsCore` is heterogeneous structure-of-arrays storage. `PhysObjAdvanced`
allows custom model adapters to inspect or replace that raw storage. Normal
particle code uses `PhysObj` typed accessors and handles `ParticleStateError`.

`Interaction<T>` and `NeighborList` are generic engines. Physical networks and
`ParticleNeighborList` intentionally hide them; model users work with particle
pairs and model-owned errors instead.

## Structured Matrices

Structured matrix types encode storage constraints that do not fit the
universal dense/sparse `Matrix<T>` contract. They are advanced because their
operations and result types depend on the represented structure. They are not
alternate backends for the basic `Matrix<T>` type.

## Extension Rules

Advanced APIs may expose representation or generic-engine concepts, but they
must not create a second normal route for model construction, arithmetic,
serialization, randomness, or thread-pool ownership. Direct Serde remains the
only conversion API, and `set_max_threads` remains the only PiP-wide execution
setting.

# PiP Advanced API

> **Alpha API:** this guide targets `4.0.0-alpha.2`. Advanced interfaces may
> change between alpha releases and have no compatibility contract with 3.x.

The advanced API is opt-in:

```rust
use physics_in_parallel::prelude::advanced::*;
```

Use `prelude::basic` or `prelude::models` first. Use this layer only for missing
coverage or a measured efficiency requirement. PiP's crate-private
implementation modules remain inaccessible, and the advanced prelude does not
implicitly import either normal prelude.

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
`for_each_stored_entry` streams through a callback without allocating in these
containers and remains usable through a trait object. Sparse order is unspecified;
callers needing ordered accumulation must sort or use the basic reductions.
The ordinary coordinate-based methods should be considered first.

Backend choice matters here:

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
pairs and model-owned errors instead. Neighbor storage tracks occupied cells;
`num_cells` is the logical grid size. Geometry capacity and nonfinite-position
errors are explicit. Particle callers normally use `rebuild_and_collect_into`;
separate queries require unchanged positions since rebuilding and distances
remain nonperiodic. See [PERFORMANCE.md](PERFORMANCE.md) for costs and freshness.

## Structured Matrices

Structured matrix types encode storage constraints that do not fit the
universal dense/sparse `Matrix<T>` contract. They are advanced because their
operations and result types depend on the represented structure. They are not
alternate backends for the basic `Matrix<T>` type. Diagonal matrix-vector work
is O(n) for finite input, using internal floating SIMD dispatch where available.
Nonfinite inputs use logical traversal to preserve implicit-zero products.

## Extension Rules

Advanced APIs may expose representation or generic-engine concepts, but they
must not create a second normal route for model construction, arithmetic,
serialization, randomness, or thread-pool ownership. Direct Serde remains the
only conversion API, and `set_max_threads` remains the only PiP-wide execution
setting.

## Refactor compatibility notes

This pass retains container types, explicit backend selection, RNG coordinates,
serialized formats, and caller-owned pools. It adds reusable output/observation
helpers and defaulted extension hooks. SIMD widths and dispatch stay internal.
`NeighborListError` adds capacity and nonfinite-position variants; downstream
exhaustive matches on that advanced error must handle them. Deserializing an
unsupported indexed RNG method, constructing unusable boundary geometry, and
querying an unbuilt particle neighbor list now fail explicitly. Force accumulation
rounding and ordered kinetic observations are documented in the performance guide.

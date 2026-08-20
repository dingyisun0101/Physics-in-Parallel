# PiP advanced API

This document is the exhaustive contract for
`physics_in_parallel::prelude::advanced`. It is opt-in and does not import the
basic or models preludes.

Advanced APIs are intended for custom algorithms, storage backends, schema
integration, and generic engines. Ordinary simulation code should prefer the
basic and models APIs in the main README.

## Exported symbols

The advanced prelude exports exactly these symbols:

- Tensor backends and layout: `Backend`, `Dense`, `Sparse`, `RowMajorLayout`,
  `TensorTrait`, `TensorResult`, `RankNDense`, `RankNSparse`.
- Matrix backends and structured matrices: `MatrixBackend`, `SparseMatrix`,
  `DiagonalMatrix`, `SymmetricMatrix`, `AntiSymmetricMatrix`,
  `UpperTriangularMatrix`, `StrictUpperTriangularMatrix`,
  `LowerTriangularMatrix`, `StrictLowerTriangularMatrix`.
- Dynamic and generated vector batches: `DynVectorList`, `VectorListRand`,
  `HaarVectors`, `NNVectors`.
- Random-fill extension points: `TensorRandElement`, `NUM_RNGS`.
- Scalar and interchange support: `ScalarSerde`, `NdarrayConvert`,
  `TensorStringConvert`, `FlatPayload`, `SparsePayload`, `SparsePayloadParts`,
  `ToJsonPayload`, `FromJsonPayload`, `JSON_SCHEMA_VERSION`.
- Generic system storage: `AttrId`, `AttrsMeta`, `AttrsCore`,
  `PhysObjAdvanced`.
- Generic interactions: `ObjId`, `InteractionId`, `InteractionNodes`,
  `InteractionOrder`, `InteractionTopology`, `Interaction`,
  `InteractionError`, `NeighborList`, `NeighborListError`.
- Generic reduction: `Reducer`, `MeanReducer`.
- Spatial extension points: `Space`, `SquareLatticeAdvanced`, `Kernel`,
  `NearestNeighborKernel`, `PowerLawKernel`, `UniformDistanceKernel`,
  `save_square_lattice`.

All public variants and documented methods on these items are part of the
advanced contract. Generated Rustdoc is the exact signature reference.

## Invariants and intended use

### Backends

`Backend`, `TensorTrait`, and `MatrixBackend` exist for algorithms that must be
generic over storage. `Dense` and `Sparse` are marker backends;
`RowMajorLayout` owns coordinate/flat-index conversion. Sparse values must be
canonical: sorted unique flat indices, in bounds, with explicit zeros omitted.

Structured matrix markers encode mathematical storage constraints. Operations
that cannot preserve such a structure return a dense result explicitly.

### Random extensions

`TensorRandElement` is the sealed set of scalar types PiP can fill directly.
It is exposed only so generic advanced code can state the same bound as
`TensorRandFiller`; downstream crates cannot implement it. `NUM_RNGS` is the
fixed deterministic lane count for stateful random streams. It is deliberately
independent of the basic API's process-wide execution partition limit, so
changing execution parallelism does not change seeded results.

`VectorListRand`, `HaarVectors`, and `NNVectors` expose reusable generators for
custom algorithms. Basic consumers should use `TensorRandFiller` or the
high-level sampling and pairing APIs.

### Generic object engine

`AttrsCore` is heterogeneous structure-of-arrays storage with stable attribute
IDs and equal object counts across columns. `AttrsMeta` stores collection
metadata. `PhysObjAdvanced` provides raw metadata and attribute-store access
when the typed `PhysObj` facade does not cover an integration need.

`InteractionTopology` canonicalizes node order according to
`InteractionOrder`; `Interaction<T>` attaches payloads to that topology.
`NeighborList` is model-agnostic spatial candidate generation. Ready particle
models wrap these facilities and belong to `prelude::models`.

### Space extensions

`Space` is the generic indexing/storage trait. Normal square-lattice code uses
the inherent lattice methods instead. `SquareLatticeAdvanced` currently adds
downsampling.

`Kernel` and its concrete implementations are exposed for custom pairing
algorithms. Normal pair generation selects built-ins through `KernelType`, so
users do not construct or dispatch kernel objects themselves.

### Interchange

Payload and conversion traits support custom persistence layers. The schema
version and payload types describe PiP data only; they do not prescribe file
layout, checkpointing, or workflow behavior.

## Internal boundary

Modules not reachable through `prelude::basic`, `prelude::models`, or this
advanced prelude are crate-private. They are free to change and must not be
duplicated in downstream code. If a missing capability can be composed from
the published layers, composition is preferred over exposing another backend
primitive.

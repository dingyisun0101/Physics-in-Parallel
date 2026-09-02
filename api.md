# PiP Public API Review

This document records the public API exposed by PiP on the `tmp` branch after
Stage 4, identifies overlapping routes, and audits communication across module
boundaries. It is a review document, not an accepted redesign. Stage 5 should
not begin until the open choices here are resolved.

## Scope and source of truth

The externally reachable modules are:

- `physics_in_parallel::prelude::basic`
- `physics_in_parallel::prelude::models`
- `physics_in_parallel::prelude::advanced`
- `physics_in_parallel::threading`

The three prelude source files are the authoritative named export lists. Public
enum variants, trait methods, inherent methods, Serde implementations, and
operator implementations on those exported types are also public API, even
when their implementation modules are crate-private.

The current documentation has two known inventory mismatches:

- `README.md` omits `SpringRecord` and `PowerLawRecord` from the models API.
- `advanced_api.md` omits the advanced re-export of
  `DEFAULT_RANDOM_MAX_THREADS`.

## Current basic API

### Scalars and containers

- `Scalar`
- `Complex`
- `ScalarCastError`
- `Tensor`
- `TensorError`
- `Matrix`
- `DenseMatrix`
- `MatrixError`
- `VectorList`

`Tensor` supplies construction, shape and rank inspection, coordinate and flat
indexing, mutation, mapping, zipping, filling, reductions, elementwise
operations, transpose, Hermitian transpose, dot products, norms, cross and
wedge products, matrix multiplication, dense/sparse conversion, scalar casts,
contiguous access, and Serde.

`Matrix` supplies dense and sparse construction, shape inspection, coordinate
and flat indexing, mutation, matrix-vector and batched matrix-vector products,
elementwise operations, scalar multiplication, transpose, Hermitian transpose,
matrix multiplication, trace, absolute value, casts, backend conversion, and
Serde.

`VectorList` supplies construction, shape inspection, scalar, vector, and axis
access, mutation, filling, parallel row traversal, row scaling, norms,
normalization, polar decomposition, elementwise operations, scalar operations,
casts, operators, and Serde.

### Randomness

- `RngMethod`
- `RngConfig`
- `RngConfigError`
- `IndexedRng`
- `RandType`
- `TensorRandFiller`
- `TensorRandError`
- `DEFAULT_RANDOM_MAX_THREADS`

`RngMethod` exposes `name`, `version`, `seed_encoding`, and `from_name`.
`RngConfig` exposes `new`, `seed`, `method`, `encode_seed`, and `resolve_for`.
`IndexedRng` exposes `new`, `try_from_resolved`, `rng_config`, `unit_f64`,
`uniform_index`, and `standard_normal`.

`TensorRandFiller` exposes panicking and fallible stateful constructors, an
indexed constructor, stateful and indexed fill operations, distribution and
RNG inspection, and an instance-local parallelism limit.

### Continuous space and sampling

- `BoundaryError`
- `ContinuousBoundary`
- `PeriodicBox`
- `ClampBox`
- `ReflectBox`
- `VectorSamplingMethod`
- `VectorSamplingError`
- `sample_vectors`

`ContinuousBoundary` supplies one-vector and flat-batch operations, with and
without velocity handling. Each box exposes validated construction plus
`dim`, `min`, and `max` accessors.

### Discrete space and pairing

- `BoundaryCondition`
- `SquareLatticeConfig`
- `SquareLatticeConfigError`
- `SquareLattice`
- `SquareLatticeInitMethod`
- `SourceMode`
- `PairingMethod`
- `PairGeneratorConfig`
- `PairGenerator`
- `KernelType`
- `KernelError`
- `PairGenerationError`

`SquareLatticeConfig` supplies validated general construction, periodic and
reflective shortcuts, geometry inspection, coordinate conversion, neighbor
lookup, and a Laplacian operation. `SquareLattice` supplies construction,
configuration and RNG inspection, coordinate and flat access, mutation, and
filling.

`PairGeneratorConfig` supplies general, independent-uniform, and kernel
constructors plus accessors. `PairGenerator` supplies construction, explicit
sweep refresh, configuration and generated-sweep inspection, and batch or
individual access to sources, displacements, and targets.

### Execution

- `ComputePool`
- `ComputePoolError`
- `with_threads`

`ComputePool` supplies construction, thread-count inspection, and installation
of an operation. `with_threads` is the one-operation convenience equivalent.
These names are also reachable directly through `physics_in_parallel::threading`.

## Current models API

### State and canonical attributes

- `PhysObj`
- `AttrsError`
- `create_template`
- `MassiveParticlesError`
- `VelocitySamplingMethod`
- `randomize_r`
- `randomize_v`
- `ATTR_R`
- `ATTR_V`
- `ATTR_A`
- `ATTR_M`
- `ATTR_M_INV`
- `ATTR_ALIVE`
- `ATTR_RIGID`
- `ParticleSelection`
- `set_alive`
- `is_alive`
- `set_rigid`
- `is_rigid`

`PhysObj` supplies empty construction, metadata inspection, object-count and
attribute-presence inspection, typed attribute and attribute-vector access,
typed mutation, serialization, and JSON file persistence.

### Boundaries, integration, thermostats, and observation

- `ParticleBoundary`
- `ParticleBoundaryError`
- `Integrator`
- `IntegratorError`
- `ExplicitEuler`
- `SemiImplicitEuler`
- `Thermostat`
- `ThermostatError`
- `LangevinThermostat`
- `Observer`
- `ObserveError`
- `KineticEnergyObserver`
- `TemperatureObserver`

The four behavior traits expose `apply_to_particles`, `apply`, `apply`, and
`observe`, respectively. `LangevinThermostat` supplies fresh and restored
construction plus physical parameter, resolved RNG, step-counter, and particle
selection accessors. Both observers supply `new`, `Default`, and a selection
accessor while also exposing selection as a public field.

### Laws and interaction networks

- `Spring`
- `SpringCutoff`
- `SpringLawError`
- `SpringRecord`
- `SpringNetwork`
- `SpringNetworkError`
- `PowerLawDecay`
- `PowerLawRange`
- `PowerLawError`
- `PowerLawRecord`
- `PowerLawNetwork`
- `PowerLawNetworkError`

Both laws expose public parameter fields, validated construction, validation,
and validated Serde.

Both networks expose empty and capacity-aware construction, strict record
restoration, record export, borrowed iteration, size inspection, raw-parameter
insertion, payload insertion, bulk insertion, removal, immutable and mutable
lookup, clearing, immutable and mutable generic-backend access, force
application, and Serde. `PowerLawNetwork` additionally exposes all-to-all
capacity calculation, empty all-to-all construction, and all-to-all insertion.

### Particle neighbor lists

- `ParticleNeighborList`
- `ParticleNeighborListError`

`ParticleNeighborList` exposes construction from bounds or box dimensions,
geometry inspection, immutable and mutable generic candidate-list access,
rebuilding, cutoff-filtered pair collection, and an exact alias for that
collection operation.

## Current advanced API

### Tensor and matrix backends

- `Backend`
- `Dense`
- `Sparse`
- `RowMajorLayout`
- `TensorTrait`
- `TensorResult`
- `MatrixBackend`
- `RankNDense`
- `RankNSparse`
- `SparseMatrix`
- `DiagonalMatrix`
- `SymmetricMatrix`
- `AntiSymmetricMatrix`
- `UpperTriangularMatrix`
- `StrictUpperTriangularMatrix`
- `LowerTriangularMatrix`
- `StrictLowerTriangularMatrix`

`TensorTrait` exposes backend construction and primitives together with nearly
the entire user-facing tensor algebra. `MatrixBackend` exposes construction,
shape, scalar access, contiguous access, sparse entry extraction, and filling.
`RowMajorLayout` exposes checked and panicking construction, shape and stride
inspection, index conversion, and coordinate conversion.

### Dynamic vectors and random extensions

- `DynVectorList`
- `VectorListRand`
- `HaarVectors`
- `NNVectors`
- `TensorRandElement`
- `NUM_RNGS`
- `DEFAULT_RANDOM_MAX_THREADS`

`DynVectorList` exposes runtime type inspection, shape inspection, cloning,
downcasting, scalar-kind inspection, and serialization. `VectorListRand`
exposes stateful construction and refresh. Both generators additionally expose
indexed construction and refresh, resolved RNG inspection, and JSON helpers;
their generated vectors, dimension, and count are public fields.

### Generic storage, interactions, and neighbors

- `AttrId`
- `AttrsMeta`
- `AttrsCore`
- `PhysObjAdvanced`
- `ObjId`
- `InteractionId`
- `InteractionNodes`
- `InteractionOrder`
- `InteractionTopology`
- `Interaction`
- `InteractionError`
- `NeighborList`
- `NeighborListError`

`AttrsCore` exposes allocation, insertion, removal, renaming, label and ID
lookup, typed access by label or ID, simultaneous mutable borrows, per-object
vector operations, type and dimension inspection, and serialization.
`PhysObjAdvanced` exposes raw construction and raw metadata and attribute-store
access.

`InteractionTopology` exposes topology construction, order and object-bound
mutation, capacity management, pruning, insertion, lookup, removal, iteration,
and pair-specific shortcuts. `Interaction<T>` repeats those operations while
attaching payloads and adds mutable and parallel payload traversal.
`NeighborList` exposes construction, geometry and occupancy inspection,
rebuilding, candidate traversal, and candidate collection.

### Sampling, reduction, and kernels

- `DynamicWeightedIndex`
- `DynamicWeightedIndexError`
- `Reducer`
- `MeanReducer`
- `Kernel`
- `NearestNeighborKernel`
- `PowerLawKernel`
- `UniformDistanceKernel`

`DynamicWeightedIndex` exposes construction, size and total inspection, weight
lookup and mutation, and selection with or without one excluded item. `Reducer`
exposes batch reduction. `Kernel` exposes indexed scalar and batch sampling,
kind inspection, and trait-object cloning. Concrete kernels expose both checked
and panicking constructors.

### Interchange and spatial extensions

- `ScalarSerde`
- `TensorStringConvert`
- `ToJsonPayload`
- `FromJsonPayload`
- `FlatPayload`
- `SparsePayload`
- `SparsePayloadParts`
- `JSON_SCHEMA_VERSION`
- `Space`
- `SquareLatticeAdvanced`
- `save_square_lattice`

The interchange traits and payloads duplicate portions of direct Serde and
inherent JSON support. `Space` exposes storage, shape, coordinate access,
mutation, filling, and file persistence. `SquareLatticeAdvanced` exposes
downsampling.

## Public error surface

The current normal API exposes these errors:

- Math and randomness: `ScalarCastError`, `TensorError`, `MatrixError`,
  `TensorRandError`, `RngConfigError`.
- Space and pairing: `BoundaryError`, `VectorSamplingError`,
  `SquareLatticeConfigError`, `KernelError`, `PairGenerationError`.
- Execution: `ComputePoolError`.
- Particle state and behavior: `AttrsError`, `MassiveParticlesError`,
  `ParticleBoundaryError`, `IntegratorError`, `ThermostatError`, `ObserveError`.
- Laws and interactions: `SpringLawError`, `PowerLawError`,
  `SpringNetworkError`, `PowerLawNetworkError`, `ParticleNeighborListError`.

The advanced API additionally exposes `InteractionError`, `NeighborListError`,
and `DynamicWeightedIndexError`.

Particle shape and count failures are duplicated across most model error enums.
Backend errors are also embedded in model error variants, which makes advanced
types part of the effective model contract. A shared public particle-state
error could remove repetition. Model wrappers should translate backend errors
that are meaningful at the model boundary and treat impossible backend failures
as invariant violations rather than exposing the backend enum wholesale.

`ThermostatError::InvalidParam` currently combines non-negative constructor
parameters, strictly positive inverse mass, and finite derived sigma under one
message. These should be separate errors or carry an explicit validation rule.

## Exact overlaps to remove

1. `ParticleNeighborList::collect_pairs_within_cutoff` is an exact alias for
   `collect_pairs`; keep only one.
2. `DEFAULT_RANDOM_MAX_THREADS` is exported by both basic and advanced; assign
   it to one tier.
3. Observer selection is both a public field and an accessor; make the field
   private.
4. `new`/`try_new`, `from_vec`/`try_from_vec`, and
   `cast_to`/`try_cast_to` pairs compete. User-controlled data should have one
   fallible route. A conventional `new` or `from_vec` may return `Result`.
5. Direct Serde, `serialize`, `serialize_value`, `to_json_*`,
   `TensorStringConvert`, payload traits, and file-saving methods provide
   overlapping conversion and persistence routes. Keep direct validated Serde;
   make payload records private; leave filesystem operations downstream.
6. `SpringNetwork::add_spring` duplicates payload construction plus
   `add_spring_payload`; retain insertion of a validated `Spring` only. Apply
   the same rule to `PowerLawNetwork::add_power_law` and `add_payload`.
7. Public network `records` and `from_records` overlap direct network Serde.
   Keep the record types private unless they gain an independent computational
   purpose.
8. Network `get_*_mut` methods and public law fields bypass validation. Make law
   fields private and replace mutation with validated whole-value replacement.
9. Network `interaction` and `interaction_mut` expose a second, generic
   mutation system beneath the model API. Remove both.
10. `ParticleNeighborList::candidates` and `candidates_mut` expose the generic
    neighbor engine beneath the model API. Remove both and expose particle-level
    traversal if it is needed.
11. `DenseMatrix<T>` is exactly the default `Matrix<T>` specialization; one
    public name is sufficient.
12. `VectorList` exposes named elementwise operations and equivalent operator
    implementations. Choose one notation for each operation.

## Responsibilities that belong downstream

PiP should expose physical and numerical behavior, while scientific-workflow
owns workflow execution and data movement. On that basis, the following APIs
should be removed or moved out of the normal PiP surface:

- `Reducer` and `MeanReducer`: aggregation across observations belongs to the
  workflow or analysis layer.
- `Space::save`, `save_square_lattice`, and `PhysObj::save_to_json`: persistence
  policy and file placement belong downstream.
- `ComputePool` and `with_threads`: outer execution scheduling belongs to the
  workflow. PiP may continue using the caller's current Rayon pool internally.

`PairGeneratorConfig` is a pure constructor DTO and conflicts with the decision
that downstream crates own configuration. `SquareLatticeConfig` should either
become an independently meaningful geometry value with a domain name or be
replaced by ordinary constructor parameters. `RngConfig` requires a separate
decision because it currently represents both an unresolved request and
resolved reproducibility state.

## Non-overlapping ownership target

Literal absence of shared concepts is neither possible nor useful. Shape and
index access, for example, naturally occur on several containers. The target
should instead be one authoritative construction, mutation, traversal, and
serialization route per abstraction layer:

- `Tensor` owns N-dimensional storage and tensor algebra.
- `Matrix` owns rank-two linear algebra, not duplicate tensor algebra.
- `VectorList` owns batched geometric-vector operations.
- `Interaction<T>` owns generic topology and payload storage.
- Physical networks own validated laws and do not reveal `Interaction<T>`.
- `ContinuousBoundary` owns geometric boundary operations.
- `ParticleBoundary` adapts a boundary to canonical particle state.
- `NeighborList` owns generic spatial candidates.
- `ParticleNeighborList` owns particle-state validation and physical cutoff
  filtering without exposing the candidate engine.
- Serde owns data interchange; downstream code owns files and workflow state.

## Module boundary policy

The API tier rule applies to communication across domain boundaries. A module
may freely use its own private children as implementation details. When it
needs another domain, it follows this order:

1. Use that domain's public basic facade, such as `crate::math`, `crate::space`,
   or `crate::rng`.
2. Use an advanced API only when the basic facade lacks required coverage or
   when avoiding it would impose a demonstrated performance cost.
3. Use another domain's crate-private implementation only as a documented last
   resort.

Internal implementation code should not import `prelude::*`. Preludes are
external conveniences that flatten ownership and can create circular facade
dependencies. Internal code should import a named domain facade instead.

Advanced and internal dependencies must remain encapsulated. A normal model
method must not return an advanced type, accept one solely for convenience, or
expose an advanced error variant. Otherwise downstream users must understand
both layers and the model wrapper no longer owns its invariants.

## Intended dependency direction

```text
rng + threading
       |
      math
       |
sampling + space + engines
       |
     models
       |
external consumers
```

`prelude::basic`, `prelude::models`, and `prelude::advanced` assemble exports;
they are not implementation layers. The advanced facade spans extension points
from several implementation layers, so lower modules may need the underlying
domain type rather than importing `crate::advanced` and creating a cycle.

## Current boundary audit

### Basic-first dependencies already in good shape

- `engines::soa::phys_obj` primarily consumes the public `crate::math` facade.
- Square-lattice representation already uses public `Tensor` and `TensorError`.
- Kernel implementations consume the canonical RNG types.
- Model law payloads are independent domain values.
- Prelude modules and `advanced.rs` primarily assemble exports rather than
  implement algorithms.

### Avoidable private cross-domain imports

- `models::particles::create_state` imports private rank-two vector-list and
  tensor modules even though `VectorList`, `RandType`, `TensorRandError`, and
  `TensorRandFiller` are available through `crate::math`.
- The same module imports vector sampling through
  `space::continuous::sampling` instead of `crate::space`.
- `models::particles::boundary` imports `ContinuousBoundary` and
  `BoundaryError` through a private path instead of `crate::space`.
- `space::continuous::sampling` reaches into private math modules instead of
  using `crate::math`.
- `models::particles::thermostat` obtains `IndexedRng` through the private
  square-lattice random module. The canonical source is `crate::rng`.

These imports have no identified coverage or efficiency justification and
should use their dependency's basic facade.

### Justified advanced dependencies

- Spring and power-law networks use `Interaction<T>` for generic topology,
  stable payload slots, and efficient parallel traversal.
- `ParticleNeighborList` uses `NeighborList` for model-independent spatial
  candidate generation.
- Pair generation uses `RowMajorLayout`, `HaarVectors`, and `NNVectors` for
  capabilities not present in the basic API.
- Math and space kernels use private threading heuristics to avoid parallel
  overhead and control work partitioning.
- Rank-two math uses rank-N storage internally for shared algorithms and
  zero-copy representation.

These dependencies are justified inside implementations. They do not justify
exposing the backend through a normal model API.

### Last-resort dependencies requiring review

- Pair generation uses the private rank-N dense storage type. Verify whether
  public `Tensor` can provide the same storage and performance. Retain the
  private type only if a measured or structural limitation remains.
- Square-lattice initialization imports private top-level indexed shuffle code.
  If the helper is lattice-specific, move it under the lattice implementation;
  if it is generally reusable, promote it to a deliberate advanced API.
- `PhysObj` depends on private math JSON infrastructure. Removing bespoke JSON
  APIs removes this edge. Otherwise, shared interchange code needs a neutral
  lower-level owner.
- `Space::save` calls private space IO. Removing persistence from the space
  contract removes both the dependency and a workflow-policy concern.

### Advanced and internal types leaking through models

- `SpringNetwork::interaction` and `interaction_mut` expose
  `Interaction<Spring>`.
- `PowerLawNetwork::interaction` and `interaction_mut` expose
  `Interaction<PowerLawDecay>`.
- `ParticleNeighborList::candidates` and `candidates_mut` expose
  `NeighborList`.
- Model error enums embed `InteractionError` and `NeighborListError`.
- Model insertion methods return advanced `InteractionId` values.
- `PhysObj` is presented as canonical model state but is owned by the generic
  engine implementation module.

The first five leaks should be removed or replaced with model-owned concepts.
The ownership of `PhysObj` should move to the models domain while `AttrsCore`
remains its advanced storage implementation.

## Enforcement

The boundary can be checked mechanically with an architecture test or lint
script that rejects unapproved cross-domain paths, including:

- `crate::math::tensor::rank_*` outside the math domain
- `crate::space::continuous::*` and `crate::space::discrete::*` outside the
  space domain
- `crate::engines::*` outside engines and explicitly approved model adapters

Each allowlisted advanced or private dependency should state whether it exists
for coverage, zero-copy representation, or measured efficiency. Tests should
also compile the basic and models preludes independently so normal APIs cannot
silently acquire advanced import requirements.

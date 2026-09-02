# PiP Public API Review

This document records the public API exposed by PiP on the `tmp` branch after
Stage 4, the clean-slate public contract approved during the API review, and
the required communication boundaries between modules. The current inventory
is retained as migration evidence; **Approved target contract** is normative
for Stage 5 and later work. No backward compatibility is required.

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

## Approved target contract

### Canonical exports and API tiers

Domain roots are the authoritative public paths. For example, math types come
from `physics_in_parallel::math`, RNG types from
`physics_in_parallel::rng`, and particle models from
`physics_in_parallel::models`.

`prelude::basic` is a small convenience set containing the normal
backend-agnostic numerical API. Advanced APIs require an explicit advanced
import. Internal implementation APIs are not externally reachable. Legacy and
overlapping re-export paths are removed.

PiP itself uses another domain's basic facade first. It may cross into an
advanced API only for missing coverage or a demonstrated efficiency need, and
may use a private cross-domain API only as a documented last resort.

### Backend-agnostic tensor family

The purpose of PiP's tensor unit is a backend-agnostic user experience with
optimized dense and sparse execution behind one interface. The canonical
basic types are:

```rust
Tensor<T>
Matrix<T>
VectorList<T>
```

Dense and sparse are both fundamental basic capabilities, but they are not
different mathematical Rust types. Each canonical container owns a private
tagged dense-or-sparse representation. Public backend implementation traits
and backend marker type parameters are removed. Downstream custom backends are
not supported; PiP may add another built-in representation without changing
ordinary function signatures.

Users choose the initial representation through structurally distinct
constructors. Dense values and sparse entries must not be forced through one
weakly typed constructor:

```rust
let dense: Tensor<f64> = Tensor::from_dense(shape, values)?;
let sparse: Tensor<f64> = Tensor::from_sparse_entries(shape, entries)?;
```

Constructors with a natural representation choose it deterministically:

- zero and identity construction use sparse storage
- filled and function-generated construction use dense storage
- no constructor guesses from a runtime density threshold

The basic `StorageKind` enum contains `Dense` and `Sparse`. `storage_kind`,
`make_dense`, and `make_sparse` are basic controls. PiP never silently changes
an existing value's representation. An allocating operation preserves its
receiver's representation, including for mixed-representation inputs. An
`*_into` operation writes into the caller-selected representation of its
output. Operand order can therefore affect storage and performance but must
never affect mathematical values. Direct Serde preserves the selected
representation and validated deserialization restores it without conversion.

Every allocating operation must contain a prominent `# Result storage`
section. Every potentially representation-sensitive operation must contain a
`# Complexity` section that states dense, sparse, and mixed costs. Operations
that can produce dense occupancy in sparse storage must place that warning
near the method summary as well as in the detailed complexity section.

### Arithmetic, access, and traversal

Named fallible methods are the sole arithmetic API. Use names such as `add`,
`multiply`, `matmul`, and their justified `*_into` forms. Remove overlapping
arithmetic operator implementations because operators hide allocation,
representation choice, validation failures, and complexity.

Basic element access is semantic and container-specific:

- `Tensor` accepts multidimensional coordinates
- `Matrix` accepts row and column coordinates
- `VectorList` accepts vector and component coordinates

`get` returns a scalar value rather than a storage reference, and `set` is
fallible. Raw accessors remain advanced. Terminology is strict throughout
code, errors, and documentation: **coordinates** identify a multidimensional
logical location; **index** means only a flattened storage position used by a
raw accessor. Matrix `row` and `column`, and vector-list `vector` and
`component`, are coordinate names rather than flat indices.

Basic `values()` traverses every logical element in coordinate order and
yields scalar values for both representations. Its sparse complexity is
`O(total logical elements)`, not `O(stored elements)`, and must be stated next
to the method. Advanced, explicitly named stored-entry traversal supplies the
`O(stored elements)` route. Contiguous slices, sparse entry mutation, flat
indexing, and other storage-native views are advanced.

### Parallel execution

Built-in mathematical and physical operations select sequential or parallel
kernels internally. PiP does not publish duplicate `foo` and `par_foo`
operations. Explicit parallel traversal is advanced and exists only for
custom callbacks that the semantic API cannot express.

PiP uses the active Rayon pool and never owns a public pool wrapper. Remove
`ComputePool`, `ComputePoolError`, and `with_threads`. Callers must configure a
shared Rayon pool themselves or let scientific-workflow coordinate it. The
crate startup documentation and every potentially parallel API must warn that
independent pools can oversubscribe the machine.

A process-wide PiP limit controls the maximum worker participation requested
by one operation:

```rust
pub fn set_max_threads(limit: Option<usize>) -> Result<(), ParallelismError>;
pub fn max_threads() -> Option<usize>;
```

`None`, which is the default, means no PiP-specific cap; the active Rayon pool
and useful work still bound execution. `Some(0)` is invalid. Each operation
snapshots the setting when it begins, and nested kernels share that operation's
budget. The setting does not create, resize, or reserve a Rayon pool. Remove
per-struct maximum-thread controls and `DEFAULT_RANDOM_MAX_THREADS`. Normal
applications call the setter once during startup before concurrent work begins.

### Randomness

PiP owns one resolved RNG value:

```rust
pub struct ResolvedRng {
    seed: u64,
    method: RngMethod,
}
```

`ResolvedRng::new(seed, method)` is deterministic. Entropy requires an
explicit `ResolvedRng::from_entropy(method)` call. There are no optional seed
or method fields, hidden defaults, entropy fallbacks, or public `RngConfig`.
Every public stochastic PiP API accepts `ResolvedRng`; no stochastic API takes
a raw seed and method as separate arguments.

Components validate supported methods once, retain `ResolvedRng` for
inspection and serialization where state is independently meaningful, and
lower it internally to indexed or stateful generators. One-shot operations do
not return the resolved input again. Long-lived stochastic state stores the
resolved RNG plus the indexed counter or stateful cursor required to resume
exactly.

PiP remains unaware of scientific-workflow. A downstream adapter, owned by
OmniFluid in this project, combines a Workflow purpose-derived `u64` seed with
an OmniFluid-selected `RngMethod` to construct `ResolvedRng`.

### Domain values and configuration ownership

PiP constructors accept ordinary semantic values, not serialization DTOs.
Remove `PairGeneratorConfig`; `PairGenerator::new` accepts shape, pairing
method, pair count, and `ResolvedRng` directly. Keep `PairingMethod`,
`KernelType`, and `SourceMode` because they are domain concepts.

Rename and reframe `SquareLatticeConfig` as an independently useful geometry
value such as `SquareLatticeGeometry`. It owns shape, spacing, boundary,
layout, coordinate, and neighbor geometry. The live `SquareLattice<T>` owns
cells and initialization provenance. It is not a downstream configuration DTO.

PiP does not define a config struct for every runtime type. OmniFluid and other
downstream crates own grouped, serializable configuration and feed validated
values through PiP accessors and constructors.

### Laws and interaction networks

`Spring` and `PowerLawDecay` have private fields, validated constructors, and
read-only accessors. Network APIs never expose mutable law references; changing
a law uses validated whole-value replacement.

Particle pairs `(usize, usize)` are the public identity of model interactions.
`InteractionId` and backend slot identities are not part of model APIs.
Networks are sparse edge collections and do not own a fixed particle count:

- `new()` creates an empty unbounded network
- `with_capacity(edge_capacity)` reserves edges only
- valid insertion automatically grows the minimum required particle bound
- insertion validates the complete request before mutation
- applying a network validates every endpoint against the actual `PhysObj`

The old `num_particles` concept is removed. A read-only
`minimum_particle_count` may expose the greatest required endpoint bound.
Self-pairs, invalid laws, arithmetic overflow, malformed batches, and
out-of-range application are model-owned errors.

Duplicate runtime insertion is an upsert:

```rust
fn insert(
    &mut self,
    pair: (usize, usize),
    law: Law,
) -> Result<Option<Law>, NetworkError>;
```

`None` means a new edge and `Some(previous)` means replacement. Bulk insertion
is transactional. Deserialization is stricter and rejects duplicate pairs so
persisted input cannot depend on record ordering.

Direct validated Serde is the sole network interchange route. Record DTOs are
private implementation details; remove public `SpringRecord`,
`PowerLawRecord`, `records`, and `from_records`. Ordinary borrowed iteration
provides inspection. Remove public `interaction`, `interaction_mut`, and all
generic-backend exposure from model networks.

`ParticleNeighborList` likewise hides `NeighborList` and exposes only
particle-level construction, rebuild, and pair traversal. Keep one name for
cutoff-filtered pair collection.

### Serialization and downstream responsibilities

Direct validated Serde is the only conversion contract. Remove bespoke string,
JSON, payload, and file persistence helpers. Payload DTOs remain private unless
they gain an independent computational purpose. Paths, formats, recording
cadence, observation aggregation, and filesystem policy belong downstream.

Remove `Reducer` and `MeanReducer`; aggregation belongs to
scientific-workflow or an analysis layer. PiP state may implement Serde where
serialization is independently meaningful, but PiP does not know Workflow's
schema or persistence types.

For every user-controlled value there is one fallible construction or
conversion route. Remove competing panicking/fallible pairs such as
`new`/`try_new`, `from_vec`/`try_from_vec`, and `cast_to`/`try_cast_to`.
Conventional names may return `Result`. An operation is infallible only when
failure is impossible for every accepted input.

### Error and trait contracts

`ParticleStateError` is the authoritative public error for invalid canonical
`PhysObj` state: missing attributes, wrong scalar types, invalid shapes,
object-count mismatches, and particle-coordinate bounds. Each model still has
its own public error type and wraps `ParticleStateError` as a source when a
state failure is discovered. Integrators, boundaries, thermostats, observers,
networks, and neighbor lists retain their genuinely domain-specific variants.

Model errors expose only model and particle-state concepts. Tensor storage,
attribute storage, generic interaction, and generic neighbor errors are
translated at the model boundary. A broken internal assumption becomes an
explicit model-owned invariant error rather than a leaked backend enum or an
unexplained panic.

All public errors implement `std::error::Error + Send + Sync + 'static` and
preserve meaningful source chains. Parameter validation uses specific variants
and values rather than catch-all strings.

Mutable behavior traits such as `Integrator` and `Thermostat` require `Send`.
Immutable behavior traits require `Send + Sync` only when PiP may share them
across workers. Public concrete values generally implement `Debug` and
`Clone`; small immutable values may implement `Copy`. Persistable state uses
direct Serde. Behavior traits do not require `Default`, `Clone`, or Serde.

## Current public error surface

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

Particle shape and count failures are duplicated across most current model
error enums. Backend errors are also embedded in model error variants, which
makes advanced types part of the effective model contract. The approved target
replaces this duplication with `ParticleStateError` and model-owned error
translation as specified above.

`ThermostatError::InvalidParam` currently combines non-negative constructor
parameters, strictly positive inverse mass, and finite derived sigma under one
message. These should be separate errors or carry an explicit validation rule.

## Approved removals and consolidations

1. `ParticleNeighborList::collect_pairs_within_cutoff` is an exact alias for
   `collect_pairs`; keep only one.
2. Remove `DEFAULT_RANDOM_MAX_THREADS` and all per-struct thread limits in
   favor of process-wide `set_max_threads` and `max_threads`.
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
11. Remove `DenseMatrix`, `SparseMatrix`, backend marker parameters, and public
    backend implementation traits. `Matrix<T>` is the universal type.
12. Remove arithmetic operator implementations from `Tensor`, `Matrix`, and
    `VectorList`; named fallible operations are canonical.
13. Remove public flat indexing and raw storage access from the basic API.
    Coordinate-based value access is basic and raw access is advanced.
14. Remove `RngConfig`; every stochastic API accepts `ResolvedRng`.
15. Remove sequential/parallel operation pairs. Built-in operations select a
    kernel internally and custom explicit parallel traversal is advanced.

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

`PairGeneratorConfig` is a pure constructor DTO and is removed because
downstream crates own configuration. `SquareLatticeConfig` becomes an
independently meaningful geometry value. `RngConfig` is replaced by the fully
specified `ResolvedRng` domain value.

## Non-overlapping ownership target

Literal absence of shared concepts is neither possible nor useful. Shape and
index access, for example, naturally occur on several containers. The target
should instead be one authoritative construction, mutation, traversal, and
serialization route per abstraction layer:

- `Tensor` owns backend-agnostic N-dimensional storage and tensor algebra.
- `Matrix` owns backend-agnostic rank-two linear algebra, not duplicate tensor
  algebra.
- `VectorList` owns backend-agnostic batched geometric-vector operations.
- Private or advanced interaction storage owns generic topology and payload
  layout.
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

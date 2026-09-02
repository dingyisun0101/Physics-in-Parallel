# PiP Public API Contract

This document is the normative clean-slate contract for PiP 4.0. No backward
compatibility with earlier releases is required.

## Ownership And Paths

The authoritative public roots are:

- `physics_in_parallel::math`
- `physics_in_parallel::rng`
- `physics_in_parallel::space`
- `physics_in_parallel::models`
- `physics_in_parallel::threading`
- `physics_in_parallel::advanced`

Implementation child modules are private. `prelude::basic`, `prelude::models`,
and `prelude::advanced` are independent convenience export lists, not alternate
ownership layers.

PiP uses another domain's basic root first. An advanced API is used only for
missing coverage or a demonstrated efficiency need. A crate-private
cross-domain API is the last resort and must have a documented reason.

## Basic Exports

`prelude::basic` exports:

- Scalars: `Scalar`, `Complex`, `ScalarCastError`.
- Universal containers: `Tensor`, `TensorError`, `Matrix`, `MatrixError`,
  `VectorList`, `VectorListError`, `StorageKind`, `Values`.
- Randomness: `ResolvedRng`, `RngMethod`, `RngError`, `IndexedRng`, `RandType`,
  `TensorRandFiller`, `TensorRandError`.
- Continuous space: `BoundaryError`, `ContinuousBoundary`, `ClampBox`,
  `PeriodicBox`, `ReflectBox`, `VectorSamplingMethod`, `VectorSamplingError`,
  `sample_vectors`.
- Discrete space: `BoundaryCondition`, `SquareLatticeGeometry`,
  `SquareLatticeGeometryError`, `SquareLattice`, `SquareLatticeInitMethod`,
  `PairGenerator`, `PairingMethod`, `SourceMode`, `KernelType`, `KernelError`,
  `PairGenerationError`.
- Execution policy: `ParallelismError`, `set_max_threads`, `max_threads`.

## Model Exports

`prelude::models` exports:

- State: `PhysObj`, `ParticleStateError`, `create_template`,
  `MassiveParticlesError`, `VelocitySamplingMethod`, `randomize_r`,
  `randomize_v`.
- Attributes and selection: `ATTR_R`, `ATTR_V`, `ATTR_A`, `ATTR_M`,
  `ATTR_M_INV`, `ATTR_ALIVE`, `ATTR_RIGID`, `ParticleSelection`, `set_alive`,
  `is_alive`, `set_rigid`, `is_rigid`.
- Boundaries: `ParticleBoundary`, `ParticleBoundaryError`.
- Integration: `Integrator`, `IntegratorError`, `ExplicitEuler`,
  `SemiImplicitEuler`.
- Thermostats: `Thermostat`, `ThermostatError`, `LangevinThermostat`.
- Observation: `Observer`, `ObserveError`, `KineticEnergyObserver`,
  `TemperatureObserver`.
- Laws: `Spring`, `SpringCutoff`, `SpringLawError`, `PowerLawDecay`,
  `PowerLawRange`, `PowerLawError`.
- Interactions: `SpringNetwork`, `SpringNetworkError`, `PowerLawNetwork`,
  `PowerLawNetworkError`, `ParticleNeighborList`,
  `ParticleNeighborListError`.

The advanced export list and its cost model are maintained in
[advanced_api.md](advanced_api.md).

## Universal Math Contract

The only basic tensor-family types are:

```rust
Tensor<T>
Matrix<T>
VectorList<T>
```

Each owns a private dense-or-sparse representation. Dense and sparse are both
fundamental built-in capabilities, not public type parameters. Downstream
custom backends are outside the contract.

Users choose representation with structurally distinct constructors. Zero and
identity construction select sparse storage; filled and function construction
select dense storage. PiP never guesses from density and never switches an
existing value implicitly.

Allocating operations preserve the receiver's representation. `*_into`
operations preserve the output's representation. Mixed-representation operand
order can affect cost and result storage, but never mathematical values. Direct
Serde records and restores representation.

Basic access is semantic and strict:

- tensors use multidimensional coordinates
- matrices use row and column coordinates
- vector lists use vector and component coordinates
- `get` returns a scalar value
- `set` is fallible

The term **index** means only a flat storage position exposed through advanced
raw access. Basic `values()` visits every logical value in coordinate order and
therefore costs `O(total logical elements)` for sparse as well as dense storage.

Named fallible methods such as `add`, `multiply`, and `matmul` are the sole
arithmetic API. Operator overloads are not provided because they hide
allocation, validation, representation choice, and complexity.

## Parallel Execution

Built-in operations choose sequential or parallel kernels internally. PiP uses
the active Rayon pool and never creates or owns a pool.

Applications must configure one shared pool or let their workflow system do so.
Independent pools can oversubscribe the machine. Normal applications call:

```rust
set_max_threads(Some(limit))?;
```

once during startup. `None`, the default, removes the PiP-specific cap.
`Some(0)` is invalid. The setting limits worker participation requested by one
PiP method; available work and the active Rayon pool remain additional limits.
It does not create, resize, or reserve threads. No object has an independent
thread setting, and there are no public pool wrappers or duplicate `par_*`
operations.

## Randomness

Every public stochastic API accepts `ResolvedRng`, containing one explicit
`u64` seed and one `RngMethod`. Deterministic construction uses
`ResolvedRng::new`; host entropy requires an explicit
`ResolvedRng::from_entropy` call. There are no unresolved RNG configs, optional
seeds, hidden defaults, or raw seed/method argument pairs.

Components validate supported methods and lower the resolved value to indexed
or stateful generators internally. One-shot operations return only their
scientific result. Independently meaningful long-lived stochastic state retains
the resolved RNG and the counter or cursor needed to continue reproducibly.

PiP does not know about scientific-workflow. A downstream application adapts a
workflow purpose-derived seed and its chosen `RngMethod` into `ResolvedRng`.

## Domain Configuration

PiP constructors accept semantic values, not serialization DTOs. In particular:

- `PairGenerator::new` accepts shape, method, pair count, and `ResolvedRng`
- `SquareLatticeGeometry` owns validated shape, spacing, and boundary geometry
- a live `SquareLattice<T>` owns cells and initialization provenance

Downstream crates own grouped user-facing configuration and lower validated
fields through PiP constructors and accessors. PiP does not define one config
struct per runtime type and does not depend on a workflow crate.

## Models And Networks

`ParticleStateError` is authoritative for missing attributes, wrong scalar
types, invalid shapes, inconsistent object counts, and particle bounds. Each
behavior retains a model-specific error and wraps particle-state failures as a
source. Public model errors expose model concepts, not tensor, attribute,
interaction-slot, or neighbor-engine details.

`Spring` and `PowerLawDecay` have private fields, fallible validated
constructors, and read-only accessors. Networks use canonical particle pairs as
identity and expose no generic interaction engine or mutable law references.

Network rules:

- `new()` creates an empty network
- `with_capacity(edge_capacity)` reserves edges only
- insertion validates before mutation and grows the required endpoint bound
- insertion is an upsert and returns the previous law
- batch insertion is transactional and rejects duplicate input pairs
- applying validates endpoints against the actual particle state
- deserialization rejects duplicate pairs
- record DTOs are private

`ParticleNeighborList` similarly hides its generic candidate engine and offers
only particle-level construction, rebuild, and pair collection.

Mutable behavior traits such as `Integrator` and `Thermostat` require `Send`.
Immutable behavior traits require `Send + Sync` when PiP may share them across
workers. Public errors implement `Error + Send + Sync + 'static`. Concrete
values generally implement `Debug` and `Clone`; small immutable values may also
implement `Copy`.

## Serialization

Direct validated Serde is the only conversion contract. Payload records remain
private. PiP has no JSON-string, text conversion, file persistence, reducer,
checkpoint, observation cadence, or output-directory API. Those responsibilities
belong to applications, workflow systems, and analysis layers.

## Removed Concepts

The 4.0 API intentionally has no public `RngConfig`, `PairGeneratorConfig`,
`SquareLatticeConfig`, `ComputePool`, backend marker parameter, dense/sparse
matrix alias, arithmetic operator overlap, model-level interaction ID, public
network record, reducer, file writer, or bespoke conversion helper.

# Physics in Parallel

> **Alpha and breaking API notice:** PiP is under active design. Version 3.4.0
> introduces a new API contract; every version before 3.4.0 is considered broken
> and unsupported. Versions still increase normally, but any alpha release may
> contain breaking changes. Pin the exact version used by a scientific project.

Physics in Parallel (PiP) provides reusable, multicore numerical building
blocks for simulations: tensors, matrices, vector batches, spatial sampling,
lattices, pair generation, and ready-to-use particle models.

## API contract

PiP deliberately has three non-overlapping preludes:

```rust
use physics_in_parallel::prelude::basic::*;    // foundational user API
use physics_in_parallel::prelude::models::*;   // ready physical models
use physics_in_parallel::prelude::advanced::*; // backend and extension API
```

Import only the layers a program uses. No prelude imports another. The old
crate-wide and domain-specific preludes no longer exist.

The `basic` and `models` symbols listed below are the complete normal user API.
The opt-in backend API is catalogued in [advanced_api.md](advanced_api.md).
Anything not re-exported by one of these preludes is implementation detail and
cannot be imported by downstream crates.

Some foundational container types have inherent methods whose signatures use
advanced-only backend types. Those methods are advanced API despite living on
the same Rust type; the signature makes the boundary explicit.

PiP itself follows the same dependency rule: compose the basic API first, use
the advanced API only when it supplies a necessary capability, and access an
internal implementation only when a public route would materially harm
performance or cannot express the operation.

## Basic API

`physics_in_parallel::prelude::basic` exports exactly these symbols:

- Randomness: `RngConfig`, `RngMethod`, `RngConfigError`, `IndexedRng`.
- Internal execution policy: `set_parallel_partitions`, `parallel_partitions`,
  `DEFAULT_PARALLEL_PARTITIONS`, `ParallelismError`.
- Scalars: `Scalar`, `Complex`, `ScalarCastError`.
- Dense mathematics: `Tensor`, `TensorError`, `Matrix`, `DenseMatrix`,
  `MatrixError`, `VectorList`.
- Random filling: `RandType`, `TensorRandFiller`, `TensorRandError`.
- Continuous space: `BoundaryError`, `ContinuousBoundary`, `ClampBox`,
  `PeriodicBox`, `ReflectBox`, `VectorSamplingMethod`, `VectorSamplingError`,
  `sample_vectors`.
- Square lattices: `BoundaryCondition`, `SquareLatticeConfig`,
  `SquareLatticeConfigError`, `SquareLattice`, `SquareLatticeInitMethod`.
- Pair generation: `PairGeneratorConfig`, `PairGenerator`, `PairingMethod`,
  `SourceMode`, `KernelType`, `KernelError`, `PairGenerationError`.

The core container conventions are consistent:

- `Tensor` uses a row-major shape and signed coordinates. Coordinate and flat
  accessors both wrap periodically, including negative and oversized indices.
- `Matrix` uses signed `(row, column)` and flat access with periodic wrapping.
  Arithmetic keeps the backend when that is mathematically valid. Scalar
  multiplication is named `scalar_mul`; `mul_vectors_into` applies one matrix
  to a caller-owned contiguous vector batch.
- `VectorList` stores `num_vectors` rows of dimension `dim`; use `vector`,
  `vector_mut`, `vector_owned`, `set_vector`, `axis`, and `set_axis`
  for row/axis access.
- `SquareLattice` exposes coordinate and flat forms of `get`, `get_mut`, and
  `set`, plus `shape`, `rank`, `num_sites`, and `fill`, without requiring the
  advanced `Space` trait. Every accessor applies its configured boundary.

Dense `Tensor`, `Matrix`, and `VectorList` provide constructors, shape/access
methods, elementwise arithmetic, `scalar_mul`, casts, and Serde. `Tensor`
additionally provides mapping and tensor algebra such as transpose, dot, norm,
cross, wedge, and matrix multiplication. `Matrix` additionally provides
transpose, Hermitian transpose, trace, matrix multiplication, and `abs`.
`VectorList` additionally provides parallel row traversal, per-row scaling,
norms, normalization, and polar decomposition. Rustdoc is the exact signature
reference.

### Randomness

`RngConfig` is the only randomness configuration accepted by normal stochastic
APIs. A missing seed uses host entropy; a missing method uses the component's
documented default. Long-lived generators expose their resolved configuration.
Indexed filling and pair generation are deterministic independently of Rayon
scheduling and PiP's execution partition setting.

### Internal parallelism

PiP uses the current Rayon pool; it never creates or owns another thread pool.
Its low-level operations create at most eight partitions by default so a task
can coexist with higher-level parallel tasks. A program may set a different
process-wide limit once, before the first PiP parallel operation:

```rust
use physics_in_parallel::prelude::basic::*;

set_parallel_partitions(4).expect("configure PiP before its first parallel operation");
```

The limit controls PiP work partitioning, not Rayon worker count. Higher-level
workflow concurrency remains the responsibility of the caller. The value is
frozen after first use; later attempts return `ParallelismError`.

### Pair generation

Construct a `PairGeneratorConfig`, then a `PairGenerator` from the lattice
shape. Every pairing method produces the same meanings and shapes: one source
site and one target site per pair. `PairingMethod::IndependentUniform` samples
source and target independently over the full shape and allows self-pairs.
Kernel-based methods instead sample a displacement. Their raw target
coordinates can lie outside the declared shape and must be passed to
`SquareLattice` access methods, whose indexing rules resolve the boundary.
Users do not add boundary logic to the generator.

## Models API

`physics_in_parallel::prelude::models` exports exactly these symbols:

- State: `PhysObj`, `AttrsError`, `create_template`, `MassiveParticlesError`,
  `VelocitySamplingMethod`, `randomize_r`, `randomize_v`.
- Canonical attributes: `ATTR_R`, `ATTR_V`, `ATTR_A`, `ATTR_M`, `ATTR_M_INV`,
  `ATTR_ALIVE`, `ATTR_RIGID`, `ParticleSelection`, `set_alive`, `is_alive`,
  `set_rigid`, `is_rigid`.
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

`PhysObj` is the canonical system state, so users do not need a wrapper struct.
Its normal API exposes metadata, object count, typed attribute access, typed
attribute-vector access, mutation, serialization, and JSON persistence.
Boolean state is accessed through the bool-facing helpers; its compact numeric
encoding is internal.

## Serialization

Scientific containers implement Serde directly. Dense data is row-major;
sparse data uses canonical sorted flat indices; heterogeneous particle
attributes preserve stable IDs and scalar types. Invalid shapes, scalar
mismatches, malformed sparse entries, unsupported schema versions, and
non-finite JSON numbers are rejected during reconstruction.

PiP owns numerical data formats, not workflow policy. Checkpoint cadence,
directories, queues, run metadata, and resumption belong to the consuming
workflow crate.

## Examples

See [EXAMPLES.md](EXAMPLES.md). Run an example with:

```bash
cargo run --release --example <example_name> -- <arguments>
```

## Development contract

- Higher layers reuse lower-layer APIs instead of duplicating algorithms.
- Inferable information is not requested again at downstream call sites.
- Backend and layout details stay out of normal user code.
- New public functionality is composed from existing APIs whenever possible.
- Advanced additions require a concrete use case not covered by the basic or
  models API.

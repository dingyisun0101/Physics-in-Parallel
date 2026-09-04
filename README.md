# Physics in Parallel

> **ALPHA / BREAKING UPDATE**
>
> `4.0.0-alpha.2` is an unstable, clean-slate rewrite. It is not compatible
> with PiP 3.x. Public APIs and serialized representations may change without
> migration support before 4.0.0. Pin the exact alpha version and do not treat
> alpha data formats as archival.

Physics in Parallel (PiP) provides reusable multicore numerical building
blocks for simulations: backend-agnostic tensors, spatial sampling, lattices,
pair generation, and ready-to-use particle models.

## Installation

During the alpha, pin the exact release:

```toml
[dependencies]
physics_in_parallel = "=4.0.0-alpha.2"
```

PiP requires Rust 1.97 or newer.

## Breaking Changes From 3.x

PiP 4.0 intentionally provides no compatibility aliases, migration wrappers,
or readers for 3.x data. The main changes are:

- `Tensor<T>`, `Matrix<T>`, and `VectorList<T>` are backend-agnostic public
  types. Every constructor receives an explicit `Backend::Dense` or
  `Backend::Sparse` selection.
- Random operations take `ResolvedRng`; optional seeds and implicit entropy
  fallbacks are gone.
- Numerical foundations and ready-made physical models use independent
  `prelude::basic` and `prelude::models` imports.
- One process-wide `set_max_threads` setting replaces per-object worker
  controls. PiP uses the caller's active Rayon pool.
- Direct validated Serde is the conversion boundary. PiP no longer owns file,
  output-directory, reducer, checkpoint, or JSON-string policy.
- PiP constructors accept semantic runtime values. Applications own grouped
  configuration types and workflow adapters.

> **Thread-pool setup is the application's responsibility.** PiP uses the
> active Rayon pool and never creates a pool. Configure one shared pool before
> starting concurrent work, especially when using a workflow runner, or the
> process may oversubscribe the machine. Call `set_max_threads` once at startup
> to cap worker participation by any single PiP method.

## API Layers

Domain roots are authoritative: `math`, `rng`, `space`, `models`, and
`threading`. Three independent preludes are available for convenience:

```rust
use physics_in_parallel::prelude::basic::*;
use physics_in_parallel::prelude::models::*;
use physics_in_parallel::prelude::advanced::*;
```

The basic prelude contains numerical, random, spatial, and execution tools. The
models prelude contains particle state and physical behavior. The advanced
prelude is opt-in and exposes raw storage and generic engines. No prelude
imports another, and domain roots remain available for explicit imports.

## Quick Start

```rust
use physics_in_parallel::prelude::basic::{
    ResolvedRng, RngMethod, VectorSamplingMethod, set_max_threads,
};
use physics_in_parallel::prelude::models::{
    ParticleSelection, Spring, SpringNetwork, create_template, randomize_r,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_max_threads(Some(4))?;

    let mut particles = create_template(2, 2)?;
    let rng = ResolvedRng::new(42, RngMethod::IndexedSplitMix64);
    randomize_r(
        &mut particles,
        VectorSamplingMethod::Uniform { low: -1.0, high: 1.0 },
        rng,
    )?;

    let mut network = SpringNetwork::new();
    network.insert((0, 1), Spring::new(1.0, 0.5, None)?)?;
    network.apply(&mut particles, ParticleSelection::AliveOnly)?;
    Ok(())
}
```

Run the complete example with `cargo run --example basic_particle`.

## Universal Containers

`Tensor<T>`, `Matrix<T>`, and `VectorList<T>` each hide a dense or sparse
implementation behind one public mathematical type. Both backends are
fundamental, and every constructor accepts the same explicit `Backend`
parameter:

```rust
use physics_in_parallel::math::{Backend, Tensor};

fn construct() -> Result<(), Box<dyn std::error::Error>> {
    let dense = Tensor::<f64>::zeros(&[64, 64], Backend::Dense)?;
    let sparse = Tensor::<f64>::zeros(&[64, 64], Backend::Sparse)?;

    let mut empty = Tensor::<f64>::empty(&[2], Backend::Dense)?;
    empty.set(&[0], 1.0)?;
    empty.set(&[1], 2.0)?;
    let initialized = empty.finish()?;
    Ok(())
}
```

`empty` returns a safe builder. Dense builders must initialize every logical
element before `finish`; unwritten sparse coordinates are implicit zeros.

- `backend` reports the current choice and `set_backend` converts explicitly.
- PiP never changes backend implicitly.
- Allocating operations preserve the receiver's backend.
- `*_into` operations preserve the caller-selected output backend.
- Direct Serde preserves the backend.
- Basic access uses coordinates and returns values.
- Flat indices, contiguous slices, and stored-entry traversal require the
  advanced `RawStorage` trait.

The sparse backend does not make every operation sparse-cost. Logical `values()`,
general mapping, and operations that produce many nonzero values can require
time or memory proportional to the full logical size. Read each method's
`# Complexity` and `# Result backend` sections before choosing a backend.

## Randomness

Every public stochastic API accepts a fully specified `ResolvedRng`. PiP has no
optional seed, hidden default method, or implicit entropy fallback.

```rust
let reproducible = ResolvedRng::new(7, RngMethod::ChaCha12);
let nondeterministic = ResolvedRng::from_entropy(RngMethod::ChaCha12);
```

Indexed operations are schedule-independent. Long-lived stochastic objects
retain the resolved RNG and any counter needed for reproducible continuation.
When PiP is used with a workflow system, the application adapts the workflow's
purpose-derived seed into `ResolvedRng`; PiP does not depend on or know about
that workflow.

## Models

`PhysObj` is canonical particle state. Typed attribute access returns universal
`VectorList` values, and invalid state is reported through
`ParticleStateError`. Integrators and thermostats require `Send`; immutable
behaviors that may be shared across workers require `Send + Sync`.

`Spring` and `PowerLawDecay` are validated immutable values. Their networks use
particle pairs as identity, grow the minimum endpoint bound automatically, and
return the previous law on replacement. Bulk insertion is transactional.
Network deserialization rejects duplicate pairs.

## Serialization

Direct validated Serde is PiP's only conversion contract. PiP does not provide
JSON-string helpers, payload APIs, file writers, reducers, output directories,
or checkpoint schedules. Applications and workflow systems own persistence and
aggregation policy.

API details live in the crate's rustdoc. Read the
[advanced API guide](https://github.com/dingyisun0101/Physics-in-Parallel/blob/main/advanced_api.md)
before using lower-level facilities, and see the
[example guide](https://github.com/dingyisun0101/Physics-in-Parallel/blob/main/EXAMPLES.md)
for the checked example inventory.

## Execution and performance

See [the performance guide](PERFORMANCE.md) for SIMD dispatch (including optional
AVX-512F), numerical ordering, reusable output APIs, sparse costs, and validation.
SIMD selection is internal and never changes the public mathematical types.

## Efficient simulation loops

Run `cargo run --release --example basic_particle` for a complete loop using one
application-owned Rayon pool. It resets acceleration, accumulates forces,
integrates, applies walls and a thermostat, then observes at an explicit interval.
Use `set_mass` to maintain mass/inverse-mass consistency and `kinetic_summary`
when energy, temperature and population are needed together. Neighbor callers can
reuse their pair buffer with `rebuild_and_collect_into`.

Dense arithmetic and scaling automatically select AVX-512F or AVX2 when supported,
with portable fallbacks. No SIMD-specific user types or CPU build flags are needed.
See [PERFORMANCE.md](PERFORMANCE.md) for numerical contracts, storage costs,
benchmark commands and hardware-validation limits; [advanced_api.md](advanced_api.md)
records extension compatibility notes.

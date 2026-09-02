# Physics in Parallel

Physics in Parallel (PiP) provides reusable multicore numerical building
blocks for simulations: backend-agnostic tensors, spatial sampling, lattices,
pair generation, and ready-to-use particle models.

Version 4.0.0 is a clean-slate API. No compatibility with pre-4.0 releases is
provided.

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

The basic prelude contains ordinary numerical and spatial tools. The models
prelude contains particle state and physical behavior. The advanced prelude is
opt-in and exposes raw storage and generic engines. No prelude imports another.

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
representation behind one public mathematical type. Both representations are
fundamental; callers choose explicitly through dense-value or sparse-entry
constructors.

- `storage_kind`, `make_dense`, and `make_sparse` control representation.
- PiP never changes representation implicitly.
- Allocating operations preserve the receiver's representation.
- `*_into` operations preserve the caller-selected output representation.
- Direct Serde preserves representation.
- Basic access uses coordinates and returns values.
- Flat indices, contiguous slices, and stored-entry traversal require the
  advanced `RawStorage` trait.

Sparse storage does not make every operation sparse-cost. Logical `values()`,
general mapping, and operations that produce many nonzero values can require
time or memory proportional to the full logical size. Read each method's
`# Complexity` and `# Result storage` sections before choosing a representation.

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
purpose-derived seed into `ResolvedRng`; PiP does not depend on that workflow.

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

See [api.md](api.md) for the complete contract and
[advanced_api.md](advanced_api.md) before using lower-level facilities.

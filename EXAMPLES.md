# Examples

These examples target PiP `4.0.0-alpha.1`. They are not compatible with PiP
3.x and may change between 4.0 alpha releases.

## Basic Particle Model

`examples/basic_particle.rs` demonstrates the current normal API:

- configure PiP's process-wide per-method thread cap
- create canonical particle state
- adapt an explicit seed and method into `ResolvedRng`
- sample positions
- construct and apply a validated pair-keyed spring network
- inspect a typed universal particle attribute

Run it with:

```bash
cargo run --example basic_particle
```

PiP uses the active Rayon pool. Real applications should configure one shared
pool before concurrent work begins; `set_max_threads` caps a PiP method but does
not create or resize that pool.

The example imports numerical and spatial foundations from `prelude::basic`
and particle behavior from the independent `prelude::models`. Import
`prelude::advanced` only when a basic or model API cannot provide the required
coverage or when measurement justifies backend-sensitive access.

Performance benchmarks should live in a dedicated benchmark harness so normal
examples remain small, current, and compile-checked with the public API.

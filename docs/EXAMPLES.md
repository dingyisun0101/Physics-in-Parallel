# Examples

These examples target PiP `4.1.0-alpha`. See the [migration notes](RELEASES.md)
before adapting code written for earlier releases. Run commands from the repository root.

## Basic Particle Model

`examples/basic_particle.rs` demonstrates the current normal API:

- configure PiP's process-wide per-method thread cap
- create canonical particle state
- initialize particle positions and keep mass/inverse mass consistent
- adapt an explicit seed and method into `ResolvedRng` for Langevin noise
- construct and apply a validated pair-keyed spring network
- integrate, reflect at walls, apply a thermostat, and report kinetic summaries

Run it with:

```bash
cargo run --release --example basic_particle
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

## NumPy interchange fixtures

`examples/numpy_fixtures.rs` emits schema-v2 documents for the
[NumPy helper](NUMPY.md). Regenerate its fixtures from the repository root:

```sh
cargo run --release --example numpy_fixtures > python/fixtures.json
python3 -m unittest python/test_numpy_support.py
```

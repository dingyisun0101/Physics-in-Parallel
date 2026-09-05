# Release and migration notes

## 4.1.0-alpha — breaking alpha release

PiP 4.1 alpha continues the 4.0 alpha rewrite and replaces the 3.x API.
Public APIs and serialized representations may change between alpha releases.
This is an intentional breaking update: review constructors, extension
traits, error matches, and persisted data before changing your dependency to
`physics_in_parallel = "=4.1.0-alpha"`. Rust 1.97 or newer is required.
Pin the exact alpha version and do not treat alpha data formats as archival.

### Upgrading from 3.x

- Use universal `Tensor`, `Matrix`, and `VectorList` containers with an explicit
  `Backend::Dense` or `Backend::Sparse` constructor argument.
- Import numerical foundations from `prelude::basic` and particle models from
  `prelude::models`. Raw storage and generic engines belong to the independent
  `prelude::advanced` layer.
- Pass an explicit `ResolvedRng` to stochastic operations. PiP no longer accepts
  optional seeds or provides an implicit entropy fallback.
- Configure one application-owned Rayon pool and call `set_max_threads` at
  startup to cap participation in each PiP operation.
- Use validated Serde directly. Applications own files, checkpoints, JSON-string
  helpers, configuration, output directories, and aggregation policy.
- There are no 3.x compatibility wrappers or legacy-data readers. Convert old
  data in an application using the version that originally wrote it.

### Upgrading from the 4.0 alphas

- The schema-v2 tensor wire format from `4.0.0-alpha.2` is retained. Schema-v1
  tensor documents from `4.0.0-alpha.1` remain unsupported. The package version
  and JSON schema version are separate; do not change a document's version to 5.
- Validation rejects oversized logical index spaces, invalid indexed RNG
  methods, unusable boundary geometry, and malformed storage payloads.
  Deserializing shapes whose product exceeds `isize::MAX` now returns an error.
- Universal container decoding now preserves full-width `i128`/`u128` values
  regardless of JSON field order. Missing, null, duplicate, or crossed storage
  fields are rejected consistently.
- Advanced `NeighborListError` has capacity and nonfinite-position variants;
  update exhaustive matches. Particle neighbor queries before rebuilding fail
  explicitly, and positions must remain unchanged between rebuild and query.
- Force contributions accumulate directly into acceleration. Large kinetic
  observations use fixed reduction blocks. These changes can alter floating
  rounding; see the [numerical contracts](PERFORMANCE.md#numerical-behavior).

### Performance and validation

Release builds enable LLVM auto-vectorization at optimization level 3, with fat
LTO and one codegen unit. Consumer applications must put these profile settings
in their own workspace root. Runtime AVX2/AVX-512 dispatch retains portable
fallbacks. Fast math is not enabled.

Dense output reuse, single-pass allocating scale, borrowed particle columns,
retained neighbor buffers, and bounded parallel observations reduce memory work.
See [benchmarks](BENCHMARKS.md) for measured gains and regressions; these are not
a claim of general superiority over optimized Rust or other numerical libraries.

Dedicated [correctness tests](CORRECTNESS.md) compare basic, model, and advanced
APIs with independent references, including 31 reference tests and JSON
regressions. The [SIMD suite](SIMD.md) checks all explicit floating kernels and
can report timings with normal auto-vectorization or an opt-in scalar baseline.

Guides now live in `docs/`. The root README and crate API entry point link to
them. Temporary refactor plans and operation logs have been removed; curated
benchmark CSVs remain under `benches/results/`.

### Earlier releases and yanking

The release procedure publishes and verifies 4.1.0-alpha before yanking all earlier
versions. Yanking excludes a version from new dependency resolution; it does
not remove the crate or invalidate existing lockfiles. See
[Cargo's yank documentation](https://doc.rust-lang.org/cargo/commands/cargo-yank.html).

## Maintainer release checks

Run from the repository root with normal compiler settings:

```sh
cargo fmt --all -- --check
cargo test --locked --release --all-targets
cargo test --locked --release --doc
bash tests/run_correctness.sh
bash tests/run_simd.sh --no-autovectorize
python3 -m unittest python/test_numpy_support.py
cargo doc --locked --no-deps
cargo publish --locked --dry-run
```

Inspect the package contents, commit the complete change, push the branch and
`v4.1.0-alpha` tag, publish the crate, and confirm the new registry version is usable.
Only then yank each still-active earlier version. Include the breaking warning
in the annotated release tag and published README, and verify the final
registry state.

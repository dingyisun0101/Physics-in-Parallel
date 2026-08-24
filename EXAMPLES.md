# Examples

This directory contains runnable examples and benchmarks for PiP. Use them for
two purposes:

- learn the intended API shape by reading the `run_pip_version` or main setup
  code;
- measure whether a PiP abstraction is behaving correctly and competitively
  against a simple baseline.

For timing numbers, prefer release builds:

```bash
cargo run --release --example <example_name> -- <args>
```

Debug builds are useful for checking behavior, but their timings can be
misleading because Rust keeps many checks and skips most optimization.

## Particle Model Benchmarks

### `spring_network_benchmark`

What it is:
A full particle-model benchmark for a sparse Hooke-law spring network. It
compares a naive array-of-structs implementation against PiP's canonical
particle state, `SpringNetwork`, and `SemiImplicitEuler`.

What it does:
- builds a deterministic list of `M` unordered spring pairs;
- initializes `N` particles in 3D;
- gives every spring the same `Spring` payload;
- evolves both implementations for `steps` time steps;
- validates final PiP positions against the naive baseline.

Why it exists:
This is the reference example for a sparse pair-interaction particle model. Read
`run_pip_version` in `examples/spring_network_benchmark.rs` to see the complete
PiP procedure: create state, fill `ATTR_R`, build a capacity-aware network,
add spring payloads, apply acceleration, and integrate.

How to run:

```bash
cargo run --release --example spring_network_benchmark -- <particles> <springs> <steps>
```

Defaults:

```text
particles = 10000
springs   = 4000
steps     = 1000
```

Example:

```bash
cargo run --release --example spring_network_benchmark -- 10000 4000 100
```

How to interpret results:
- `Initiation Time` includes particle allocation, initial position writes, and
  network construction.
- `Evolution Time` is average milliseconds per time step.
- `Throughput` is particles processed per second, not spring interactions per
  second.
- `Speedup` is `Naive / PiP`; values above `1.0x` mean PiP is faster.
- `Numerical Validation` should pass with a tiny max difference. A failure means
  the two implementations no longer compute the same trajectory.

### `power_law_network_benchmark`

What it is:
A full particle-model benchmark for an all-to-all power-law interaction model.
It compares a naive array-of-structs implementation against PiP's canonical
particle state, `PowerLawNetwork`, and `SemiImplicitEuler`.

What it does:
- constructs every unordered particle pair `(i, j)` with `i < j`;
- initializes `N` particles in 3D;
- gives every pair the same `PowerLawDecay` payload;
- evolves both implementations for `steps` time steps;
- validates final PiP positions against the naive baseline.

Why it exists:
Power-law models are naturally all-to-all unless an approximation or neighbor
list is introduced. This example demonstrates that full model lifecycle. Read
`run_pip_version` in `examples/power_law_network_benchmark.rs` for the complete
PiP procedure.

How to run:

```bash
cargo run --release --example power_law_network_benchmark -- <particles> <steps>
```

Defaults:

```text
particles = 1000
steps     = 100
```

Example:

```bash
cargo run --release --example power_law_network_benchmark -- 1000 100
```

How to interpret results:
- The pair count is printed as `N * (N - 1) / 2`; runtime and memory grow
  quadratically with particle count.
- `Initiation Time` includes all-to-all network construction.
- `Evolution Time` is average milliseconds per time step.
- `Throughput` is particles per second. For this example, also keep the pair
  count in mind because pair work dominates at larger `N`.
- `Speedup` is `Naive / PiP`; values above `1.0x` mean PiP is faster.
- `Numerical Validation` should pass. A failure means the PiP and naive force
  conventions or integration order diverged.

## Tensor And Vector Benchmarks

### `tensor_rand_large_benchmark`

What it is:
A benchmark for large dense tensor random fills. It compares PiP's
`TensorRandFiller` against a sequential `Vec<f64>` fill.

What it does:
- allocates one dense `f64` tensor with `len` elements;
- fills it repeatedly from `Uniform(-1, 1)`;
- repeats the same workload sequentially for comparison;
- tests one or more RNG implementations.

How to run:

```bash
cargo run --release --example tensor_rand_large_benchmark -- <len> <repeats> <max_threads> <rng_methods>
```

Defaults:

```text
len         = 120000000
repeats     = 3
max_threads = 8
rng_methods = pcg64,pcg64mcg,smallrng
```

Examples:

```bash
cargo run --release --example tensor_rand_large_benchmark -- 10000000 3
cargo run --release --example tensor_rand_large_benchmark -- 10000000 3 8 all
cargo run --release --example tensor_rand_large_benchmark -- 10000000 3 8 pcg64,smallrng
```

How to interpret results:
- `tensor best` and `tensor avg` are PiP random-fill timings.
- `seq best` and `seq avg` are sequential baseline timings.
- `best x` and `avg x` are `sequential / tensor`; values above `1.0` mean PiP
  is faster.
- Checksums are sanity checks only. They should be finite and comparable in
  scale, not identical, because different generators and sharding produce
  different random streams.

### `vector_list_haar_benchmark`

What it is:
A benchmark for generating Haar-distributed unit vectors in `VectorList`.

What it does:
- creates `num_vecs` vectors of dimension `dim`;
- fills them with Gaussian components;
- normalizes each row to unit length;
- compares PiP's `HaarVectors` path against a sequential implementation.

How to run:

```bash
cargo run --release --example vector_list_haar_benchmark -- <dim> <num_vecs> <repeats>
```

Defaults:

```text
dim      = 3
num_vecs = 5000000
repeats  = 3
```

Example:

```bash
cargo run --release --example vector_list_haar_benchmark -- 3 5000000 3
```

How to interpret results:
- `VectorList` is the PiP implementation.
- `Sequential` is the scalar baseline.
- `speedup` is `sequential best / VectorList best`; values above `1.0` mean PiP
  is faster.
- Component sums should be small relative to `num_vecs`. For independent unit
  vectors, each component has mean zero, so the reported sums are statistical
  sanity checks.

## API Demonstrations

### `serde_flat_json`

What it is:
A demonstration of PiP's framework-independent Serde contract for core data
structures.

What it does:
- creates a dense tensor;
- creates a sparse tensor;
- creates a `VectorList`;
- creates a periodic square lattice;
- serializes each structure directly through `serde_json`; and
- prints the versioned, scalar-tagged JSON, including compact sparse indices
  and values.

How to run:

```bash
cargo run --example serde_flat_json
```

How to interpret results:
The output is intended for inspection. It shows the JSON shape used by PiP data
structures and is useful when checking persistence, interchange, or debugging
serialized state. No workflow or storage framework is involved.

## Practical Notes

- Use small arguments first to confirm correctness, then scale up in release
  mode.
- Benchmarks include setup time because model construction is part of practical
  usage.
- `Speedup` columns are always baseline divided by PiP. Values above `1.0x`
  favor PiP; values below `1.0x` favor the baseline.
- Validation checks compare model results, not performance. A benchmark with
  failed validation should be treated as incorrect regardless of timing.

# Numerical correctness against independent references

```sh
# From pip; prints results and saves the entire log
bash tests/run_correctness.sh

# Run only the dedicated integration suite
cargo test --release --test correctness -- --nocapture --test-threads=1
```

The runner executes the API reference suite with one and four Rayon workers,
then the direct scalar/AVX2/AVX-512 correctness suite. Logs are stored under
`target/correctness-results/<UTC timestamp>-<PID>.log`; override the directory
with `CORRECTNESS_LOG_DIR`. A failed assertion stops the runner with a nonzero
exit code, including when output is piped through `tee`. Ordinary `cargo test`
also discovers these integration tests.

Every `REFERENCE PASS` row reports the number of checked values, maximum
absolute error, and maximum fraction of the allowed error. A fraction at most
1 passes. A failure identifies the operation, element, actual/reference values,
absolute error, and tolerance. Discrete comparisons require exact equality.
The compiler version, build flags, and worker count are included in the log.

## Reference design

The numerical oracles operate on ordinary vectors, explicit coordinates, and
scalar formulas. They do not obtain expected values by calling another PiP
backend or the operation being tested. For example, matrix products use a
three-loop multiplication; neighbors use all unordered particle pairs;
boundaries use repeated wall crossings; interactions use a plain ordered map;
and weighted selection uses an expanded list of tickets.

The versioned indexed RNG reference independently evaluates every coordinate
without prepared prefixes or batch helpers, with a fixed known-answer word.
Distribution and particle-noise references consume these scalar reference
samples. Stateful RNG tests use serial `rand` streams and its distribution
primitives, seeded according to the documented fixed 32-lane partition. This
checks PiP's stream ownership, distribution selection, and fill traversal;
it does not independently validate the third-party random-number algorithms.

## Coverage by API level

| Level / module | Reference comparisons |
| --- | --- |
| Scalar / `math.rs` | All signed/unsigned primitive widths; integer square roots, saturation, projections, real f32/f64 operations; Complex<f32/f64> magnitude, square root, products, division, dot and Hermitian operations |
| Tensor, Matrix, VectorList / `math.rs` | Allocating and reusable add/subtract/multiply/divide, scale, maps, reductions, casts, geometry, constructors/builders, mutation, backend conversion, and Serde |
| JSON / `json.rs` | Independent versioned fixtures for every scalar type and all three containers; structured dense/sparse matrix schemas; heterogeneous attributes and slot IDs; reordered fields with full-width i128/u128 values; bitwise floating round trips; malformed schemas, duplicate fields, sparse ordering, missing/null/crossed storage arrays, nonfinite values, and index-space overflow |
| Linear algebra / `math.rs` | Tensor/Matrix products, reusable products, transposes, Hermitian transposes, trace, maximum magnitude, matrix-vector and batched products, cross/wedge, vector norms, normalization, polar decomposition, axes/rows, per-vector scaling |
| Advanced storage / `advanced.rs` | Object-safe raw storage, dense buffers, sparse entries, callbacks, heterogeneous attributes, and PhysObj adapters against vectors/maps |
| Structured matrices / `advanced.rs` | All seven structures: logical and wrapped access, dense materialization, matrix-vector/batch products, trace, maximum magnitude, addition/subtraction/scaling, dense-returning elementwise/product/transpose methods, casts, and sparse conversion; large diagonal SIMD lengths |
| Advanced engines / `advanced.rs` | Ordered/unordered interactions, replacement/removal, parallel payload mutation, pruning, higher-arity topology, and dynamically updated/excluded weighted selection |
| Spatial / `space.rs` | Periodic/reflective/Neumann index mapping, neighbors and anisotropic Laplacians; all continuous boundaries in single/batch/flip-mask/particle forms; lattice state, downsampling, and Serde |
| Neighbor search / `space.rs` | Occupied-cell candidates and every particle query form against exhaustive pairs, including motion, dead particles, coincidence and cutoff equality |
| Randomness / `random.rs` | Indexed words, unit/normal/bounded samples and rejection; indexed floating distributions; all six stateful RNGs with floating distributions, inclusive i64/isize/usize sampling, and i64 Bernoulli draws; every vector/velocity sampler, distance kernel, pairing/source mode, and lattice initialization mode |
| Particle models / `models.rs` | Both Euler orders, masks, masses, kinetic energy/temperature and observer wrappers, spring/power-law pair forces, coincident/cutoff pairs, accumulation, Langevin replay, and a coupled 100-step spring trajectory |
| Private SIMD / `simd/mod.rs` | All explicit AVX2/AVX-512 kernels against scalar formulas, including lane tails, alignment, IEEE edges, allocating chunk boundaries, and repeated Euler steps; see [SIMD.md](SIMD.md) |

Numerical comparisons focus on meaningful valid operations. Structured
operations whose results leave the represented structure use the dense-returning
API (for example a triangular transpose or an antisymmetric square). Existing
contract suites under `tests/math`, `tests/models.rs`, `tests/engines.rs`, and
`tests/execution.rs` additionally cover invalid configurations, transactional
errors, concurrency contracts, and metadata. The new suite adds independent
invalid-output preservation checks. This is a coverage inventory, not a claim
that every possible generic instantiation or error branch is exhausted.

## Tolerances and workloads

For finite results, the criterion is:

```text
abs(actual - expected) <= absolute_tolerance + relative_tolerance * abs(expected)
```

- Basic f32/f64 arithmetic uses 32 times that type's machine epsilon for both
  tolerances. f32 inputs are rounded to f32 before evaluating the reference in
  f64, so the comparison measures arithmetic error rather than input conversion.
- Sum/dot/norm reductions scale absolute tolerance by the reference L1 magnitude
  (or sum of squares), accommodating cancellation and ordered/parallel grouping.
- General f64 linear algebra, spatial and model calculations use `atol=2e-12`
  and `rtol=2e-11`. These also cover differently factored force/OU formulas and
  accumulated trajectory rounding.
- f32 matrix products use `atol=2e-4`, `rtol=2e-5`; complex f32 calculations use
  `2e-5` for both. f32 casts use `1e-6` for both. These workloads are bounded and
  deterministic; tolerances are explicit at each assertion.
- Constructors, storage, mutations, integer operations, topology, random stream
  replay, and index results use exact comparison. Nonfinite results require
  matching infinity signs or both NaN. NaN payloads are not compared. Signed
  zeros compare numerically in this suite; the direct SIMD suite checks their
  bits, and boundary flip masks also explicitly distinguish their signs.

Inputs include zeros, mixed signs, nonuniform masses, complex components, mixed
dense/sparse operands and destinations, unaligned SIMD tails, rectangular
matrices, anisotropic grids, empty selections, dead/rigid particles, and data
sizes spanning SIMD and parallel thresholds. Model reference tests also create
one- and four-worker caller pools. Seeds and input formulas are fixed, so no
statistical/flaky probability thresholds or speed assertions are needed.

JSON fixtures are built from scalar arrays and literal schema fields, then
decoded independently of PiP's serializers. Large integers are compared through
typed arrays rather than a floating-point JSON intermediate. Oversized sparse
shapes must return a deserialization error without panicking, and valid payloads
must preserve 128-bit integers regardless of JSON member order. Sparse documents
with a billion logical elements are checked through their stored entries, without
materializing the implicit zeros.

This suite measures correctness. Use `bash tests/run_simd.sh --timings` for
normal release kernel timings, or add `--no-autovectorize` for a controlled
strictly scalar reference. Neither timing mode asserts a speedup.

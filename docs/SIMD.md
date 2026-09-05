# SIMD comparisons

From `pip`, run:

```sh
bash tests/run_simd.sh             # release correctness, terminal output + saved log
bash tests/run_simd.sh --timings   # normal release timings; references may auto-vectorize
bash tests/run_simd.sh --timings --no-autovectorize  # strictly scalar baseline
```

Logs go to `target/simd-results/<UTC timestamp>-<PID>.log`. Set `SIMD_LOG_DIR`
to use another directory. Failures propagate through `tee` as a nonzero exit
status. Each operation prints a `PASS` row with the backends, case count, and
mismatch count; failures identify the operation, backend, size, input dataset,
alignment offset, and failing element. Unsupported instruction sets print
`SKIP` and are never called. A full three-way comparison requires a CPU and OS
that support AVX2 and AVX-512F.

`tests/simd/mod.rs` is included as a unit-test module by `src/math/kernels.rs`
so it can call the actual private intrinsic implementations directly. Ordinary
`cargo test` includes correctness tests. To see their output directly:

```sh
cargo test --lib math::kernels::simd:: -- --nocapture --test-threads=1
```

## Coverage

| Operation | Types | Compared implementations |
| --- | --- | --- |
| Add, subtract, multiply, divide | f32, f64 | Scalar, AVX2, AVX-512, automatic dispatch |
| In-place scale | f32, f64 | Scalar, AVX2, AVX-512, automatic dispatch |
| Copy-scale into guarded storage | f32, f64 | Scalar, AVX2, AVX-512, allocating dispatch copied into guards |
| Allocating scale | f32, f64 | Scalar, AVX2, AVX-512, actual allocating dispatch |
| Explicit and semi-implicit Euler | f64 | Scalar, AVX2, AVX-512, automatic dispatch |

These are all explicit SIMD kernels currently in `src`. Matrix/tensor/vector
container arithmetic shares the floating kernels; operations with only ordinary
Rust implementations do not have separate AVX2/AVX-512 implementations to force.
Existing integration tests continue to cover the public containers and models.

Correctness cases cover:

- Empty and short inputs, every length from 0 through 145, both dispatch
  thresholds (32 and 128), vector lane boundaries, and large odd tails.
- Input/output offsets spanning every lane alignment, with different offsets
  for binary inputs and outputs; prefix/suffix guards detect excess writes.
- All ordered pairs of 16 floating-point edge values, reproducible random bit
  patterns, and finite mixed-sign data. Edge values include signed zeros,
  subnormals, extreme finite values, infinities, and NaNs.
- Scale factors including signed zero, negative, tiny, maximal, infinite, and
  NaN values. Equality is bitwise (including zero signs), except NaNs are
  compared by classification because payload propagation may differ.
- Both Euler update orders, distinct position/velocity/acceleration data,
  positive/negative/zero/extreme/NaN time steps, and 17-step finite trajectories.
- Allocating scale around 262,144-element parallel chunk thresholds, odd chunk
  tails, and caller pools with one and four workers. No global thread cap is
  changed by these tests.

## Timing output

`--timings` runs the same correctness suite plus an otherwise ignored test.
`TIMING` CSV rows report type, operation, backend, element count, median/min/max
nanoseconds per call, nanoseconds per element, and speedup over the `Scalar`
reference. In normal release mode that reference may itself auto-vectorize. There
are three warmups and seven samples, with repeated calls per sample. Timings
use one Rayon worker and sizes 31, 128, 4,099, 262,145, and 1,048,589. Allocation
is included in `allocating_scale`; `copy_scale` measures just writes into an
existing destination and therefore has no `Auto` timing row. In-place scale
uses -1 to avoid repeated multiplication drifting into overflow/subnormals.

By default the runner preserves normal release auto-vectorization. The optional
`--no-autovectorize` flag disables both LLVM auto-vectorizers with
`-Cno-vectorize-loops -Cno-vectorize-slp`, leaving explicit target-feature
intrinsics intact. Only this mode makes `Scalar` a no-packed-SIMD baseline.
The script preserves caller flags and prints them, along with the selected mode.
Caller overrides can still disable vectorization in normal mode. Scalar floating-point
instructions still use the architecture's floating registers. The scalar
reference implements the documented formulas independently; it is not a
runtime override of production dispatch. Direct SIMD calls bypass dispatch
size thresholds, allowing both instruction sets to be tested on identical
sizes even when automatic dispatch would choose another backend.

Timing ratios are informational, with no flaky speed assertions. They include
small test-wrapper overhead and depend on CPU frequency, cache, system load,
and AVX-512 throttling. Disabling auto-vectorizers measures a deliberately
scalar baseline and can overstate gains over optimized application code. Neither
mode measures complete application performance.
Build flags and compiler version are recorded in every log.

The [recorded SIMD measurements](BENCHMARKS.md#simd-kernel-measurements) used
the old runner with auto-vectorization disabled. New default-mode timings must
be labeled separately when comparing results.

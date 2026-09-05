# Reproducible performance measurements

Large-data throughput and memory use are PiP's primary optimization targets.
Small inputs are correctness and overhead checks. Run the opt-in harness with:

```bash
cargo bench --offline --bench kernels -- --run
cargo test --offline --release --test allocation
```

The harness reports nanoseconds per complete operation as median/min/max of five
samples after warm-up. It uses one caller-owned four-worker pool with PiP caps
of one and four. Allocating operations include allocation; `*_into` reuse output.
Matrix products are checked against an ordered slice reference. The reference
is a comparison kernel, not the historical PiP implementation. All-target tests
compile the harness without executing timing workloads.

The [initial local results](../benches/results/2026-09-04-before-large-tuning.csv)
were collected on 2026-09-04 with Rust 1.97.1 / LLVM 22.1.6, default x86-64 target
flags, release optimization/LTO, and an Intel Xeon E5-2698 v3. AVX2 is available;
AVX-512 is not. Runs are not core-pinned and thermal/frequency effects are not
controlled. Treat them as local measurements, not cross-machine guarantees.

The initial pass found useful larger-matrix blocking gains and costly scheduling
for allocating scale at 100,003 values and 128×128 transpose. Those two operations
now use a more conservative 262,144-element parallel threshold; SIMD remains
independent of worker count. Million-element cases are included. Further tuning
is prioritizing data larger than the last-level cache and million-particle work.

Allocation regressions and the repeated baseline probe show:

| Operation | Review baseline requested bytes | Refactored requested bytes |
| --- | ---: | ---: |
| Dense add_into, 100,000 f64 | 800,008 | 0 |
| Dense fill, 100,000 f64 | 800,008 | 0 |
| Sparse scale, one stored / 100,000 logical | 800,156 | 156 |
| Explicit Euler, 100,000 × 3 particles | 9,800,044 | 0 |
| Independent pair refresh, 100,000 × 2 | 4,800,048 | 0 |

These are warmed single-worker requested heap bytes, not RSS or allocator
metadata. In-repository allocation tests also cover dense thermostat/boundary/
observation, random fill, graph replacement and force application.

Final linked disassembly of the dependent scale probe contains packed `vmulpd`
in both AVX2 and AVX-512F target-feature kernels. Runtime detection selects the
supported implementation. This verifies generated instructions. A later run on AVX-512 hardware also
checked execution and timings, recorded separately below.

## Large-data tuning results

Run the primary large-data profile with:

```bash
cargo bench --offline --bench kernels -- --run --large
```

It includes arrays up to 10,000,003 f64 values (80 MB per buffer), a 512×512
matrix product, a 2048×2048 lattice, one million 3D particles, and 250,000-particle
neighbor rebuild/query. These cover buffers beyond this host's last-level cache.
The same five-sample protocol and caller-owned pool are used.

[Before](../benches/results/2026-09-04-large-before.csv) and
[after](../benches/results/2026-09-04-large-after.csv) contain all raw timing ranges.
The baseline is the validated implementation delivered in `eff5fd5`, before the
large-data tuning; its public behavior is used by the same large-workload harness.
Selected four-worker medians follow:

| Complete operation | Before (ms) | After (ms) | Ratio before/after |
| --- | ---: | ---: | ---: |
| Allocating scale, 1,000,003 f64 | 1.619 | 0.392 | 4.13× |
| Allocating scale, 10,000,003 f64 | 58.600 | 21.630 | 2.71× |
| Explicit Euler, 1,000,000 × 3 | 3.437 | 2.526 | 1.36× |
| Kinetic summary, 1,000,000 × 3 | 7.798 | 2.737 | 2.85× |
| Neighbor rebuild/query, 250,000 | 211.962 | 106.347 | 1.99× |

Single-thread results are retained too: this pass does not improve every case.
For example, the million-particle Euler median is higher in the final single-worker
run. Unchanged matrix/lattice/add kernels also vary between runs, so do not assign
all timing differences to code changes. The primary gains correspond to removing
scale's full copy, eliminating per-cell neighbor allocations, and distributing
large observations. Mask scans run inside worker chunks to avoid a serial
million-particle prepass. The measurements prioritize large parallel throughput;
they are not universal speedup guarantees.

Allocating scale now initializes result memory once, with runtime SIMD and
worker-local first writes. Neighbor rebuild uses sorted contiguous records and
retained lookup capacity: warmed rebuilds allocate no heap memory, and candidate
queries allocate only two dimension-sized coordinate buffers. Kinetic scratch
is proportional to the number of fixed 16,384-particle reduction blocks, rather
than copying full particle columns. Those fixed blocks keep reduction grouping
independent of thread count. AVX-512 execution remains unverified on this host.


## SIMD kernel measurements

[Raw SIMD timing rows](../benches/results/2026-09-04-simd-scalar-baseline.csv)
were recorded on 2026-09-04 with Rust 1.97.1 on an Intel Xeon Gold 6148
(AVX2 and AVX-512F), one Rayon worker, and release optimization/LTO. This is a
separate host from the historical large-workload results above. All direct
SIMD correctness tests passed. The run used three warmups and seven timing
samples per case, reporting medians and ranges.

**Both compiler auto-vectorizers were disabled in this run.** These ratios
isolate explicit SIMD against deliberately scalar reference loops; they do not
establish gains over ordinarily optimized Rust. Reproduce that mode with:

```sh
bash tests/run_simd.sh --timings --no-autovectorize
```

| Operation | Scalar (µs) | AVX2 (µs) | AVX-512 (µs) |
| --- | ---: | ---: | ---: |
| Add 4,099 f32 | 4.024 | 0.862 | 0.763 |
| Add 262,145 f32 | 194.511 | 127.334 | 127.340 |
| Explicit Euler, 4,099 components | 4.389 | 1.372 | 1.415 |
| Allocating scale, 1,048,589 f64 | 661.878 | 665.925 | 696.522 |

Small/cache-resident arithmetic benefits substantially in these cases. AVX-512
is not consistently faster than AVX2. Large-array gains diminish, consistent
with memory traffic and allocation dominating; allocating scale here gains
nothing. Timing noise, cache, frequency, and system load remain uncontrolled.

For normal release measurements that allow the compiler to vectorize reference
loops, use `bash tests/run_simd.sh --timings`. The log states the mode and flags.
Even that kernel comparison includes test wrappers and does not measure an
entire simulation. A general advantage over alternatives still requires
comparisons with optimized Rust at equal allocation reuse and worker counts,
and end-to-end simulation workloads. The historical before/after table compares
PiP revisions, not competing numerical libraries.


## Normal release reference comparison

[Normal release timing rows](../benches/results/2026-09-05-simd-release.csv)
were collected on 2026-09-05 on the same Intel Xeon Gold 6148, with Rust 1.97.1,
fat LTO, optimization level 3, one codegen unit, and no `RUSTFLAGS` or encoded
flags. The run used the same one-worker, three-warmup, seven-sample protocol.
The measured release candidate uses the same numerical code and compiler
settings as 4.1.0-alpha. All six SIMD correctness/timing tests passed.

Here `Scalar` names the independent Rust reference, which the compiler is
allowed to auto-vectorize:

| Operation | Rust reference (µs) | AVX2 (µs) | AVX-512 (µs) | Auto (µs) |
| --- | ---: | ---: | ---: | ---: |
| Add 4,099 f32 | 1.285 | 0.852 | 0.732 | 0.739 |
| Add 262,145 f32 | 130.019 | 129.927 | 129.965 | 129.952 |
| Explicit Euler, 4,099 components | 2.661 | 1.373 | 1.418 | 1.419 |
| Allocating scale, 1,048,589 f64 | 664.416 | 674.769 | 885.710 | 701.980 |

Allowing auto-vectorization narrows the gains: automatic dispatch is about
1.74× faster for the small add and 1.88× for Euler, essentially equal for the
large add, and slower for allocating scale. The AVX-512 allocating-scale sample
range is especially wide (744–998 µs); do not infer a stable penalty from that
single median. These are kernel measurements with portable target settings,
not a comparison against native-target Rust or complete applications.

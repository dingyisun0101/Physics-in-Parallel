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

The [initial local results](benches/results/2026-09-04-before-large-tuning.csv)
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
supported implementation. This verifies generated instructions; AVX-512 execution
and throughput still require suitable hardware. No AVX-512 speedup is claimed.

# PiP review operations log

## Implementation work — authorized 2026-09-04

The user approved the complete refactor with idiomatic Rust, comprehensive documentation, and a commit/push per batch. The implementation table in `improve.md` is the current checklist; earlier sections below remain the historical review record.

- Starting branch: `main`; remote: `origin` (`dingyisun0101/Physics-in-Parallel`). Only the two review documents were untracked at the start.
- Batch 1 implements primitive integer square roots and checked derived tensor shapes, retaining negative signed-root and numeric conversion semantics. Added regressions for integer boundaries and sparse inputs with oversized derived shapes.
- Remote fetch initially failed because the system SSH configuration has invalid ownership/permissions. A per-command SSH configuration avoids that system include without modifying host configuration; the restricted environment then reported DNS failure. Requested escalated network access for the authorized remote synchronization.

This log records review actions, evidence, validation results, and concise decision summaries. It does not contain private internal reasoning.

## 2026-09-04 — Scope and discovery

- Scope: review PiP for computational efficiency and end-user ergonomics; record recommendations in `improve.md`. Implementation changes and the separate workflow review are outside this pass.
- Read the crate manifest, main README, Python helper README, and repository inventory. Checked ancestor locations and the PiP tree for `AGENTS.md`; none applies to PiP.
- The DSES parent directory is not a Git repository. PiP is an independent repository; its working tree was clean at the beginning of the review.
- Baseline: PiP `4.0.0-alpha.2`, explicit dense/sparse backends, resolved RNGs, caller-owned Rayon pool, and validated Serde. Recommendations should preserve these intentional boundaries while reducing repeated work and API friction.
- Review plan: inspect execution and allocation paths, container operations, particle stepping and interactions, spatial algorithms, serialization, and examples; run relevant existing checks and targeted disposable probes; rank findings by evidence and expected user impact.

## Source inspection and baseline checks

- Reviewed universal tensor/matrix/vector-list operations, execution-budget helpers, RNG validation, integrators, thermostat, observers, particle masks, both force networks, neighbor-list construction/traversal, pair generation, and Rust/Python tensor serialization paths.
- Baseline commit: `1a5408c` (`Unify tensor construction by backend`). Toolchain: `rustc 1.97.1`, `cargo 1.97.1`.
- `cargo test --all-targets --offline`: passed. Captured output in `/tmp/pip-review-tests.log`; reran with redirected output after the initial combined tool output was truncated.
- `cargo test --doc --offline`: passed, one doctest.
- `python -m unittest python/test_numpy_support.py`: passed, six tests.
- `cargo fmt --all -- --check`: passed.
- Initial findings to verify: universal `zip_into` replaces output storage; vector-list editing gathers and reconstructs full logical buffers; multiple public Rayon paths omit execution-budget helpers; pair refresh allocates despite its allocation-stability claim.
- Correctness checks queued: `IndexedRng` deserialization versus constructor validation; thermostat error atomicity; derived output-shape overflow; neighbor geometry capacity handling; Python scalar validation parity.
- All research so far uses the checked-out source and local execution. No external API/version claims or external-source research is needed for this repository review.

## Review order clarified

- User requested progression from fundamental modules to more advanced modules.
- Reordered the report and remaining work accordingly: scalar semantics and layout → universal containers → RNG/execution policy → interchange → spatial primitives and pairing → particle models → advanced storage/interaction engines.
- Revisited scalar implementations and scalar-algebra tests before continuing upward. Existing tests explicitly establish saturating integer magnitude/squared magnitude, integer square roots, and complex-to-real projection; those semantics are intentional and should not be silently changed during optimization.

## Targeted disposable probes

- Created a separate temporary Cargo package at `/tmp/pip-review-probe`, depending on the checked-out PiP crate. No library implementation or checked-in tests were edited.
- Command: `cargo run --release --offline --manifest-path /tmp/pip-review-probe/Cargo.toml`; output: `/tmp/pip-review-probe.log`.
- Allocation probe wraps the system allocator, counts allocation/reallocation requests and requested bytes, warms each operation once, then counts one invocation inside a one-worker Rayon pool. Counts are total allocation traffic, not peak or retained memory; inputs and object construction are excluded.
- Confirmed `Tensor::add_into` and dense `fill` each request 800,008 bytes for 100,000 `f64` elements (two allocation calls); sparse `scale` with one stored entry over the same logical size requests 800,156 bytes (four calls).
- Confirmed `ExplicitEuler::apply` at 100,000 particles × 3 dimensions requests 9,800,044 bytes (12 calls); independent pair refresh at 100,000 pairs × 2 dimensions requests 4,800,048 bytes (six calls).
- Confirmed `ReflectBox::apply_positions` on 100,000 three-dimensional positions makes 100,000 allocation calls totaling 300,000 requested bytes, even with all positions inside the box.
- Confirmed constructor/Serde mismatch: `IndexedRng::new(ResolvedRng::new(7, ChaCha12))` rejects the method, while deserializing the same resolved RNG into `IndexedRng` succeeds. The restored object reports ChaCha12 but produces the same indexed value as SplitMix64 with seed 7.
- Confirmed thermostat partial mutation on error in a one-worker pool: two one-dimensional velocities `[1, 1]`, inverse masses `[1, -1]`, target temperature 0, friction 1, and `dt=1` produce `Err` with velocities `[0.36787944117144233, 1]`; the step counter remains zero.
- Confirmed recoverable-input panics using `catch_unwind`: sparse matrix product `[2^32, 1] × [1, 2^32]`; neighbor-list construction with box `[1.0]` and cutoff `f64::MIN_POSITIVE`; bulk application of an empty-dimensional periodic box constructed from `[]`/`[]`. These probes do not allocate giant physical arrays.
- Confirmed thread-cap bypass with a public custom `ContinuousBoundary` that records Rayon worker indices. `apply_to_particles` uses four distinct workers in a four-worker pool with `set_max_threads(Some(1))`; 20,000 particles, one dimension.
- Python probes (NumPy 2.5.2) confirmed acceptance of `u128: [-1]`, `i128: ["wrong"]`, `complex_f32: [[1e100, 0]]` (converted to infinity), and nonfinite lattice values. Existing six Python tests do not cover these cases.
- Checked and dismissed an apparent network-growth concern: `InteractionTopology::set_n_objects` already returns immediately when growing, so ordinary growing insertion does not rescan all prior edges. Bulk network cloning remains a separate allocation concern.

## Scalar comparison and final synthesis

- Added a scalar microbenchmark to the temporary probe: 100,000 `u64` inputs, 2 through 100,001, five rounds, black-boxed inputs, comparing `<u64 as Scalar>::sqrt` with `u64::isqrt`. Median times were 109.995692 ms and 1.729113 ms respectively; result sums matched in each round. This is a local scalar measurement, not an end-to-end simulation speedup or exhaustive equivalence proof.
- Wrote 16 findings in `improve.md`, ordered by module level. Each records evidence, a proposed change, and acceptance criteria. Marked measured defects/costs separately from unmeasured kernel, graph, and grid optimization proposals.
- Preserved existing architectural choices in the recommendations: backend choice remains explicit; random identity remains explicit; applications own Rayon pools and workflow/persistence policy.
- Verified all local Markdown targets exist and line anchors are within their files; corrected exact function anchors during the final reference pass.
- `git status --short` shows only the two requested new documents, `improve.md` and `log.md`. No implementation, existing tests, manifest, lockfile, or workflow files were changed. Temporary probes and their build output remain under `/tmp/pip-review-probe`; captured check output remains under `/tmp/pip-review-*.log`.
- Baseline totals: 41 Rust unit/integration tests, one doctest, six Python tests; formatting check passed. Probe processes exited successfully after catching and reporting the intended panic cases. No full-suite repeat was needed after documentation-only edits.

## Compact reproduction references

The temporary allocation/worker/scalar harness can be rerun with:

```bash
cargo run --release --offline --manifest-path /tmp/pip-review-probe/Cargo.toml
```

The RNG validation mismatch can also be reproduced in any executable depending on this checkout and `serde_json`:

```rust
use physics_in_parallel::prelude::basic::*;

let wrong = ResolvedRng::new(7, RngMethod::ChaCha12);
assert!(IndexedRng::new(wrong).is_err());
let json = serde_json::to_string(&wrong).unwrap();
assert!(serde_json::from_str::<IndexedRng>(&json).is_ok()); // Current defect.
```

The Python scalar validation cases can be rerun from the PiP directory:

```python
import sys
sys.path.insert(0, "python")
from numpy_support import _payload_to_ndarray

for scalar, data in [
    ("u128", [-1]),
    ("i128", ["wrong"]),
    ("complex_f32", [[1e100, 0]]),
]:
    document = {
        "backend": "dense",
        "tensor": {
            "kind": "tensor", "version": 2, "scalar": scalar,
            "shape": [1], "data": data,
        },
    }
    print(scalar, _payload_to_ndarray(document))
```

These reproductions document current defects; they are not the desired regression-test assertions after fixes. Review deliverables are complete. The separate workflow review and implementation work remain outside this pass.

## 2026-09-04 — Follow-up SIMD scan

- User requested SIMD wherever possible and asked to record opportunities in `improve.md`. Retained the fundamental-to-advanced review order and the existing API-preservation recommendation.
- Rechecked the PiP tree for local instructions and reviewed the manifest, scalar traits, universal and older internal tensor/matrix loops, random fillers, lattice Laplacian, continuous boundaries, pair assembly, particle stepping/observables/forces, neighbor filtering, and weighted/shuffle algorithms.
- Source search found no explicit PiP SIMD intrinsics, `std::arch` imports, target-feature kernels, or runtime CPU-feature dispatch. This does not establish the absence of compiler-generated SIMD or SIMD inside dependencies.
- Local compiler: Rust 1.97.1, LLVM 22.1.6, x86_64 Linux. Host: Intel Xeon E5-2698 v3, with AVX2/FMA advertised; no AVX-512 advertised. ARM and AVX-512 execution cannot be validated on this host.
- Consulted official Rust architecture/portable-SIMD documentation and LLVM vectorization documentation for dispatch, portability, reduction, and transcendental constraints. A local compile of `std::simd::Simd::<f64, 4>::splat(1.0)` failed with E0658 (`portable_simd`), so the recommendations do not assume nightly-only APIs are available on PiP's current toolchain.
- Created `/tmp/pip-simd-probe`, a separate dependent crate with wrappers around current PiP scale, sum, and matrix-vector operations and equivalent slice reference loops. Inspecting both generic and host-targeted code generation; final linked executable disassembly is needed because library assembly emitted before final LTO may not show the final vectorization decisions.

### SIMD code-generation results

- Built final release executables with `opt-level=3`, `lto=true`, and `codegen-units=1`, first using default target features and then with `RUSTFLAGS='-C target-cpu=native'` applied to the dependent crate and dependencies. Kept exported wrappers from being inlined so their final arithmetic could be located consistently.
- Saved executables as `/tmp/pip-simd-probe/generic-bin` and `native-bin`; disassembled each wrapper using `objdump -d --no-show-raw-insn --disassemble=FUNCTION BINARY`. Outputs are `FUNCTION.generic-final.s` and `FUNCTION.native-final.s` in the probe directory.

| Inspected f64 wrapper | Generic final executable | Host-targeted final executable |
| --- | --- | --- |
| Current PiP tensor scale | Scalar `mulsd` | Scalar `vmulsd` |
| Current PiP tensor sum | Scalar `addsd` | Scalar `vaddsd` |
| Current PiP matrix-vector operation | Scalar multiplication/addition | Scalar multiplication/addition |
| Plain slice scale reference | Packed `mulpd` plus scalar tail | 256-bit `vmulpd` plus scalar tail |
| Plain slice dot reference | Some packed multiplication; scalar ordered addition | Some packed multiplication; scalar ordered addition |

- Inspected arithmetic instructions rather than treating register names or vectorized memory copies as proof of vectorized computation. The observations are limited to these concrete wrappers and build settings; they are not a whole-crate vectorization inventory.
- Executed both final binaries at lengths `1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 1003`. All 11 sizes passed in each build, comparing PiP scale/sum/matrix-vector results with slice references. These checks validate probe behavior, not all numerical edge cases or proposed SIMD implementations.
- No latency benchmark was run in this SIMD follow-up. Existing allocation and scalar-square-root measurements remain from the earlier review.

Reproduction commands (temporary package and binaries remain outside the repository):

```bash
cargo build --manifest-path /tmp/pip-simd-probe/Cargo.toml --offline --release
RUSTFLAGS='-C target-cpu=native' cargo build --manifest-path /tmp/pip-simd-probe/Cargo.toml --offline --release
objdump -d --no-show-raw-insn --disassemble=slice_scale /tmp/pip-simd-probe/generic-bin
objdump -d --no-show-raw-insn --disassemble=slice_scale /tmp/pip-simd-probe/native-bin
objdump -d --no-show-raw-insn --disassemble=pip_scale /tmp/pip-simd-probe/native-bin
```

### SIMD recommendations recorded

- Added S01–S07 to `improve.md` within their corresponding module levels: scalar semantics; dense/sparse containers and matrix kernels; indexed/stateful randomness and bounded Rayon chunks; interchange validation; lattice/boundary/pairing kernels; particle updates/observables; advanced neighbor/force/structured-matrix engines.
- Added implementation gates for internal CPU dispatch, stable-toolchain support, unaligned input/tails, numerical behavior, replay, and final-binary inspection. No public SIMD types, required vector padding, global native-CPU build flags, or new SIMD dependency were introduced.
- The scan identifies lattice interior stencils and dense particle updates in addition to container arithmetic. It explicitly separates packed arithmetic opportunities from sparse hash access, conflict-prone force writes, dependent tree updates, and shuffle swaps.
- Technical references used: [Rust architecture intrinsics and dispatch](https://doc.rust-lang.org/stable/core/arch/), [Rust portable SIMD status](https://doc.rust-lang.org/std/simd/index.html), and [LLVM vectorization guidance](https://llvm.org/docs/Vectorizers.html). Recommendations based on PiP source and local disassembly are identified separately from these platform constraints.
- Verified documentation links, line bounds, and whitespace. `git status --short` continues to show only `improve.md` and `log.md`. No library/API, dependency, manifest, existing test, or workflow changes; the earlier full-suite results were not rerun for this documentation-only follow-up.

### Batch 1 validation and delivery

- `cargo test --offline --test math` and its release counterpart: 21 tests passed in each profile.
- `cargo fmt --all`, warning-free `cargo doc --offline --no-deps`, and `git diff --check` passed.
- Escalated fetch succeeded with the per-command SSH configuration; local and remote main were identical before this batch.
- Committing scalar/layout fixes together with the approved review record and implementation checklist; pushing the batch to origin/main.

### Batch 2 — containers and optional SIMD

- Batch 1 was pushed as `bc356ec` to origin/main.
- Added reusable dense output, native sparse support traversal, ordered sparse reductions, finite sparse matrix joins, and dense matrix kernels with contiguous column blocks.
- Added private sealed scalar dispatch to stable AVX2 and AVX-512F floating kernels, with exact feature checks, ordinary slice alignment, and scalar tails. The user explicitly requested AVX-512 during this batch; it is enabled when detected for sufficiently large chunks. The host can compile but cannot execute that path.
- Added vector/axis output-copy APIs and vector-list arithmetic output APIs; matrix multiplication now has a reusable output variant.
- Added tests for allocation reuse, oversized sparse logical domains, backend combinations, rollback on shape errors, and unaligned floating SIMD edges. Added `PERFORMANCE.md` documenting dispatch, numerical behavior, allocation, and sparse cost.
- Initial new-test compilation needed an explicit f64 annotation; initial strict Clippy found two collapsible conditionals, now corrected. Existing tests passed before the new regression set.

- Final batch validation: all targets passed in debug; library, math, and allocation regressions passed in release. Strict all-target Clippy, formatting, and warning-free rustdoc passed after the final edits.
- The repeated release allocation probe reports zero allocations for dense add_into/fill; sparse single-entry scale now requests 156 bytes instead of 800,156 bytes at 100,000 logical elements. Native/PiP integer-root timings are now comparable. These are local probe results, not cross-hardware throughput guarantees.
- AVX-512F paths compile and have hardware-gated numerical tests. Execution on this AVX2-only host remains unavailable; the initial 128-element threshold needs compatible-hardware benchmarking. Public container APIs and serialized formats remain compatible.

### Batch 3 — indexed randomness and interchange

- Batch 2 pushed as `fd1bc7e`.
- IndexedRng deserialization now validates the resolved method using its constructor, retaining the transparent format. Tests cover every method.
- Hoisted seed/step/domain and item prefixes in indexed uniform/Bernoulli fills. Portable independent wrapping arithmetic is available to compiler vectorization; no AVX-512 integer-extension requirement or altered stateful stream lanes. Compared batched values to the original five-coordinate mixer across extreme seeds, split rows and odd lengths.
- Added public Python decoded-payload adapter and optional dense-result element cap. Validated 128-bit ranges/types, post-conversion complex finiteness, and untagged lattice real/complex values. Added Rust-generated fixtures and malformed variants.
- Kept JSON parsing and fallible scalar diagnostics unchanged; further explicit SIMD classification depends on profiling parsing versus validation.
- Validation passed: all Rust targets, strict all-target Clippy, warning-free rustdoc and formatting; all ten NumPy tests executed successfully (none skipped).

### Batch 4 — spatial primitives

- Batch 3 pushed as `df7926b`.
- Rejected empty and unrepresentable boundary spans using existing structured errors; reflecting geometry also validates doubled width. Reflection uses direct position/velocity updates and remainder parity, avoiding per-position masks and saturating integer crossing counts. Periodic translation handles finite overshoots whose subtraction would overflow.
- Pair generation now writes into all retained dense caches for independent, nearest-neighbor and radial methods. Warm single-worker tests count zero allocations across refresh plus bulk reflection, and verify same-sweep replay and source/displacement/target identity.
- Lattice stencils traverse contiguous per-axis interior spans with fixed offsets and retain general boundary spans. Cached inverse spacing is omitted from serialization and reconstructed by validation. Exact comparisons cover ranks 1–4, tiny extents, three boundary kinds, anisotropic spacing and component counts 1/2/3/8/17.
- An initial test imported a private implementation module; corrected it to the public space API. Spatial/allocation regressions and strict Clippy passed.
- All-target Rust tests, warning-free rustdoc, formatting and diff checks passed before delivery.

### Batch 5 — particle state and model ergonomics

- Batch 4 pushed as `e433e4a`.
- Added safe disjoint typed attribute borrows with optional borrowed flag columns, and moved successful attribute type-name formatting off the hot path. Euler updates fuse each row while preserving the explicit/implicit velocity convention. Dense updates, masks and read columns now avoid full-column copies.
- Thermostats prevalidate every included inverse mass and derived sigma before writes. Regression cases put invalid mass first/last, verify untouched velocities and counter, and verify continuation against a restored thermostat for dense/sparse backends.
- Built-in boundaries update position and velocity together. Custom boundaries conservatively stage both columns for error atomicity; the new defaulted trait method documents when an implementation can opt out of staging after dimension validation. A four-worker caller pool with a one-thread PiP cap records exactly one callback worker.
- Added kinetic_summary for one-pass energy/temperature/count, borrowing dense inputs and preserving ordered serial accumulation. Added set_mass with finite positive mass/reciprocal checks and two-field consistency. Added a complete shared-pool example with acceleration reset, forces, integration, walls, thermostat and observation.
- Warm allocation regression measures zero allocations for the dense integration/thermostat/built-in-boundary/observation/mass-update sequence. An initial regression reversed create_template's dimension/count arguments; fixed the test setup. Model, execution-budget and allocation tests passed on both supported backends.
- All targets, strict Clippy, warning-free rustdoc, formatting and the executable simulation example passed before committing.

### Batch 6 — advanced engines

- Batch 5 pushed as `e51efd1`.
- Neighbor geometry now checks axis/stride/count arithmetic and stores only occupied cells. High-rank stencils use a bounded allocation policy with occupied-cell comparison fallback. A trillion-cell logical grid with three particles is exercised without dense-grid allocation; invalid rebuilds preserve previous buckets.
- Added particle rebuild-and-query, reusable-output and callback APIs. Dense positions and flags are borrowed; pre-build/count-mismatch queries fail. Documentation requires unchanged positions for the separate fast query and explicitly states nonperiodic raw-distance semantics. Tests compare moved/dead configurations against brute force.
- Network batch insertion commits validated incoming entries without cloning the existing graph. Stack pair lookup keys avoid allocations while retaining public owned keys and serialization. Initial allocation regression exposed the payload wrapper still forwarding through allocating generic lookup; switched it to native pair lookup.
- Force paths validate all endpoint/mass errors before direct serial edge accumulation, eliminating dense input copies and full acceleration scratch. Documented the resulting floating-rounding difference from the former temporary-sum approach. Tests verify force balance, rigid selection, validation rollback, and zero allocations for dense force application plus repeated pair replacements.
- Added allocation-free, object-safe stored-entry callbacks and an O(n) diagonal matrix-vector kernel for finite inputs, with nonfinite fallback tests. Bounded the remaining Maxwell-Boltzmann postprocessing and advanced payload visits by the operation policy.
- Engine/model/allocation tests, strict Clippy and warning-free rustdoc passed. Wider parallel force accumulation and architecture-specific hash traversal remain benchmark-led alternatives, rather than unmeasured default changes.
- Final all-target tests, strict Clippy, formatting and diff checks passed before delivery.

### Batch 7 — reproducible measurements and final review

- Batch 6 pushed as `6c83353`.
- Added an opt-in dependency-free release benchmark harness covering short/odd/large f32/f64 container kernels, sparse scaling, small/rectangular/large matrix products against an ordered slice reference, transpose, lattice stencils, pairing, complete Euler updates, kinetic summaries, graph replacement, force application and neighbor rebuild/query. It reports five-sample median/min/max at one- and four-thread caps in one application-owned four-worker pool.
- Final review connected diagonal multiplication to the shared floating SIMD dispatcher and removed the remaining dense copy in public tensor random fills. Current sealed distribution implementations validate before writing; added allocation and error-atomicity checks. Added indexed fill/pair replay comparisons across caller pool sizes.
- Fixed the benchmark's initially unhandled transpose Result, and changed a new integration test from a private random-slice helper to the public fill_at API. The first complete release suite passed before those small final corrections; rerunning release verification on the final code.
- Expanded README, advanced extension notes and performance documentation, including new advanced neighbor error variants, stricter validation behavior, additive APIs and force rounding semantics.

- Final verification passed: 64 Rust tests in debug and release, one doctest, all ten Python tests, strict all-target Clippy, warning-free rustdoc, formatting, unchanged regenerated Rust/NumPy fixtures, and diff checks. The benchmark completed with all result assertions passing.
- Timing identified scheduling costs for allocating scale at 100,003 elements and 128×128 transpose; raised those operations' parallel threshold to 262,144 elements while retaining SIMD, then repeated the benchmark with million-element cases.
- Rebuilt the final linked SIMD probe; its scale implementations contain packed AVX2 and AVX-512F vmulpd. Executed the host-supported path against references. The original review probe needed its empty-boundary unwrap replaced by a constructor-error check after the intended validation fix; the updated complete probe passed and confirms the recorded allocation reductions and corrected error behavior.
- Added BENCHMARKS.md and raw local measurement CSV. User clarified that large-data performance is the design priority. Delivering the validated baseline batch before a separate large-data tuning batch concentrating on full-buffer traffic, large particle operations and occupied-cell indexing.

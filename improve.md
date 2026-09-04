# PiP efficiency and ergonomics review

Reviewed 2026-09-04 at `1a5408c` (`4.0.0-alpha.2`). The findings below describe that review baseline; implementation progress is tracked here and in [log.md](log.md). Source line links refer to the reviewed version and may move during refactoring.

## Implementation batches

The user approved implementation, documentation, and a commit/push after each batch. Preserve idiomatic Rust and the numerical/public API contracts described below.

| Batch | Scope | Status |
| --- | --- | --- |
| 1 | Scalar integer roots, cast semantics, checked derived shapes (F01/F02/S01) | Complete; validated in debug/release |
| 2 | Reusable dense/sparse container kernels, SIMD, execution budgets, traversal APIs (F03–F05/F07/S02) | Complete; runtime AVX-512F/AVX2 with portable fallback |
| 3 | Validated RNG, exact batched randomness, Python interchange (F06/F08/S03/S04) | Complete; exact uniform batching and validated adapters |
| 4 | Boundary validation/reuse, lattice stencil and pair generation (F09–F11/S05) | Complete; validated geometry and allocation-stable spatial kernels |
| 5 | Particle borrows, integration/observables, atomic thermostat, mass ergonomics (F12–F14/S06) | Complete; borrowed dense updates and validated model APIs |
| 6 | Neighbor geometry/querying, graph edits, structured/force kernels (F15/F16/S07) | Pending |
| 7 | Benchmarks, complete examples and API documentation, final validation | Pending |

The largest recurring cost is rebuilding storage inside APIs that already receive reusable objects. Fixing container traversal, mutation, and execution policy first will improve sampling, pairing, and particle stepping without forcing users onto advanced APIs.

Preserve explicit backend selection, resolved randomness, independent preludes, caller-owned Rayon pools, and application-owned persistence/workflow policy. P1 means a correctness/contract defect or a major recurring cost; P2 means a subsequent optimization or ergonomic improvement; P3 means a narrower opportunity. Measurements below establish current behavior, not promised application speedups.

**SIMD follow-up, 2026-09-04:** apply SIMD wherever it yields useful throughput while preserving the operation's numerical contract. The S01–S07 additions below follow the same fundamental-to-advanced order. Keep vector widths, CPU detection, alignment handling, and scalar tails internal; existing user code should benefit automatically. SIMD is complementary to the allocation and algorithmic improvements above. The scan found no explicit SIMD dispatch or intrinsics in PiP source; compiler-generated SIMD and dependency implementations must be assessed separately.

## 1. Scalar foundations

### F01 — P3: Replace the generic integer-square-root search with native integer operations

**Evidence:** [`isqrt_u128`](src/math/scalar/real.rs#L20) searches the full 64-bit result interval, using up to 64 iterations with `u128` division. All integer scalar implementations route through it, including small integer inputs. A local release probe over 100,000 `u64` values, 2 through 100,001, measured median times of **110.0 ms for PiP versus 1.73 ms for primitive `isqrt`**, across five rounds; result sums matched.

**Change:** use the primitive integer implementation, preserving PiP's explicit zero result for negative signed inputs. Keep saturating magnitude/squared-magnitude behavior and the documented complex-to-real projection unchanged. Clarify that `try_cast` checks conversion range; it does not promise lossless conversion.

**Acceptance:** compare individual results at zero, perfect squares and their neighbors, signed minima, and integer maxima. Benchmark representative integer workloads before treating this as a high-priority application optimization.

### S01 — Scalar operations: establish the numerical contract before vectorizing

Start with bulk `f32`/`f64` add, subtract, multiply, scale, absolute value, and square root. A single scalar trait call is not the unit of SIMD work: batch independent values inside internal slice kernels. Generic Rust functions can specialize through monomorphization; replacing the public sealed `Scalar` trait is unnecessary.

For integers, preserve the existing distinction between primitive arithmetic, explicitly wrapping RNG arithmetic, checked indices/weights, and saturating scalar magnitude operations. Hardware wrapping addition must not silently replace overflow checks or saturation. SIMD support and cost vary by integer width; retain scalar fallbacks for costly divisions and 128-bit arithmetic. F01's integer-square-root improvement remains independent of SIMD.

Complex addition, scaling, and conjugation are candidates for interleaved component processing. Complex multiplication, division, and magnitude need separate kernels and edge-case checks; their real/imaginary layout and overflow behavior cannot be replaced by real-only formulas or unchecked pointer casts. Do not silently enable FMA contraction, approximate reciprocal/square-root instructions, or altered NaN/signed-zero behavior merely to obtain packed instructions.

## 2. Layout, storage, and universal containers

### F02 — P1: Validate derived shapes before arithmetic and allocation

**Evidence:** [`Tensor::matmul`](src/math/tensor/universal.rs#L571) computes `rows * columns` before validating the result shape, then uses an unchecked constructor. [`wedge`](src/math/tensor/universal.rs#L535) has the same pattern. On this 64-bit host, multiplying sparse tensors shaped `[2^32, 1]` and `[1, 2^32]` **panics in a release build**, although each input is cheap and valid and `matmul` returns `Result`.

**Change:** compute and validate output layout once with `checked_num_elements` before forming iteration ranges. Pass the checked size through the kernel. Apply the same rule to every operation that creates a different shape, retaining signed-index-space checks.

**Acceptance:** oversized derived shapes return the existing structured shape/index errors in debug and release builds. Shape errors leave supplied output buffers untouched. The overflow case must be tested using sparse inputs so validation requires no giant allocation.

### F03 — P1: Make output-buffer and in-place operations reuse storage

**Evidence:** [`zip_into`](src/math/tensor/universal.rs#L636) collects a new full-size vector and replaces the destination. [`fill`](src/math/tensor/universal.rs#L357) and `map_in_place` also replace storage. [`VectorList::edit_values`](src/math/tensor/universal_vector_list.rs#L420) collects every logical value, invokes a closure, and reconstructs the container. Dense `add_into` and dense `fill` each request **800,008 bytes in two allocations** for 100,000 `f64` values after warm-up.

**Change:** dispatch on backend once per operation. Borrow dense slices for reads and mutations; reuse destination capacity. Keep a clearly defined sparse path and explicit scratch space where staging is necessary. Do not change error atomicity accidentally while removing copies; F13 identifies an existing failure case to repair.

**Ergonomics:** provide corresponding `VectorList` output-buffer operations and backend-independent ways to copy a vector/axis into caller storage or iterate its values. Today [`vector`/`axis`](src/math/tensor/universal_vector_list.rs#L229) return allocated vectors, while basic arithmetic offers fewer reuse options than `Tensor`/`Matrix`. Users should not need `RawStorage` for routine allocation control.

**Acceptance:** warmed dense `fill`, `map_in_place`, and elementwise `*_into` perform no heap allocations; output capacity remains reusable. Check all input/output backend combinations and shape failures. Document allocation guarantees separately from backend guarantees.

### F04 — P1: Preserve sparse computational cost where the mathematics permits it

**Evidence:** universal `scale`, `sum`, `transpose`, `cast`, and elementwise arithmetic traverse the full logical domain; several also build a dense temporary. Scaling a **single stored value in a 100,000-element sparse tensor requests 800,156 bytes**. This retains the sparse representation but loses its memory advantage during computation.

**Change:** introduce stored-entry kernels for zero-preserving transforms, sparse addition/subtraction, reductions, transpose, and appropriate mixed-backend operations. Clearing a sparse tensor to zero should clear its entries directly. General user maps and operations that change implicit zeros still need logical-domain semantics.

**Semantic constraint:** do not blindly reuse every old sparse operator. Division at implicit `0/0`, multiplication by nonfinite values, and reduction order require explicit treatment. Preserve current mathematical behavior or document an intentional API change with tests.

**Acceptance:** hold stored-entry count fixed while increasing logical size; temporary allocation and traversal for supported sparse operations should remain tied to stored entries. Compare against dense reference results, including cancellation, implicit zeros, and nonfinite cases.

### F05 — P2: Use specialized dense matrix and traversal kernels behind the same API

**Evidence:** [`matmul`](src/math/tensor/universal.rs#L571) runs a generic dot product for each output element with strided right-hand-side reads. Every scalar read goes through [`get_flat_unchecked`](src/math/tensor/universal.rs#L721), which delegates to backend accessors with periodic index normalization. Sparse reads additionally recompute/validate logical size in `len_dense`. [`mul_vectors_into`](src/math/tensor/universal_matrix.rs#L335) repeats individual matrix-vector calls and validation. Actual optimizer elimination and latency impact need measurement.

**Change:** retain validated layout metadata and dispatch once into slice-based dense kernels or stored-entry sparse kernels. Benchmark blocked dense multiplication and batched traversal before choosing an implementation. Combine Hermitian transpose/conjugation in one result pass; avoid full input copies in vector norms and normalization.

**Acceptance:** measure small and large matrices, rectangular shapes, batches, and all backend combinations. Preserve complex semantics, result backend, and documented reduction accuracy. Add `matmul_into` only with real output reuse, after F03.

### S02 — Dense containers: highest-priority SIMD work

The [`Values` iterator and flat access](src/math/tensor/universal.rs#L721) preserve backend dispatch and wrapped-access machinery inside hot loops. F03/F05 should expose contiguous slices to the optimizer before adding handwritten intrinsics. Start with these kernels:

| Kernel | SIMD direction | Required preparation or constraint |
| --- | --- | --- |
| Tensor/Matrix/VectorList elementwise arithmetic and `*_into` | Process consecutive elements in vector-width blocks. | Match backend once; validate lengths once; write directly to reusable output. Keep ordinary arbitrary-closure mapping as a fallback. |
| `fill`, scale, absolute value, square root, casts | Use slice loops and compiler vectorization first. | Filling/copying may already lower to optimized memory routines. Fallible casts must retain range checks and error behavior. |
| Vector scaling, norms, normalization | Process a block of vectors, with per-vector factors and masks. | [`VectorList`](src/math/tensor/universal_vector_list.rs#L300) stores interleaved components; dimensions 2/3 are too short to rely only on vectorizing within one vector. Remove full-buffer gathers first. |
| `sum`, dot products, norms, maximum magnitude | Vectorize independent products/classification; design accumulation separately. | Floating addition order and NaN propagation are part of correctness. Preserve order where required; any faster reordered mode must have explicit accuracy/reproducibility semantics. |
| Dense matrix multiplication | Use a blocked kernel, vectorizing over consecutive output columns with shared scalar coefficients. | Avoid strided RHS reads. Preserve each output's accumulation order initially; evaluate FMA separately. Include tile/packing cost in benchmarks. |
| Matrix-vector and batched matrix-vector multiplication | Use dense row slices, or multiple independent rows/vectors per block. | Horizontal reduction is the limiting dependency. Reuse matrix tiles across a batch rather than repeatedly validating and dispatching. |
| Transpose/Hermitian transpose | Tile loads/stores and combine conjugation with the transpose. | Measure memory throughput and shuffle cost; no extra full-size transpose/conjugate intermediate. |

**Code-generation evidence:** a separate release executable wraps current PiP operations and simple slice references. With final LTO applied, generic x86-64 slice scaling contains packed `mulpd`; the current PiP scale wrapper contains scalar `mulsd`. With `target-cpu=native`, slice scaling uses 256-bit `vmulpd`, while PiP scaling still uses scalar `vmulsd`. Current PiP sum and matrix-vector arithmetic also remain scalar in both inspected builds. The slice dot reference contains some packed multiplication but ordered scalar additions. Both executables passed reference comparisons at 11 sizes, including short and odd lengths. These observations concern the tested `f64` instantiations, not every PiP operation or every possible caller build. Reproduction commands are recorded in `log.md`; no throughput gain was measured.

**Sparse rule:** vectorize arithmetic over already-contiguous stored values or reusable packed blocks only when packing pays for itself. Hash lookup, insertion, and zero pruning remain separate costs. Never scan/materialize the full logical domain solely to obtain SIMD; F04's sparse complexity requirements take precedence.

## 3. Randomness and execution policy

### F06 — P1: Make `IndexedRng` deserialization enforce constructor invariants

**Evidence:** [`IndexedRng`](src/rng.rs#L150) derives transparent `Deserialize`, bypassing `IndexedRng::new`. Deserializing the JSON encoding of `ResolvedRng::new(7, RngMethod::ChaCha12)` succeeds, although construction with that value fails. The restored object reports ChaCha12 but produces the same indexed sample as SplitMix64 with seed 7.

**Change:** deserialize `ResolvedRng`, then call the validated constructor. Keep every method accepted by `ResolvedRng` itself; constrain only the component that cannot implement it.

**Acceptance:** every unsupported method fails through both construction and deserialization. Supported values preserve metadata and samples across a round trip. This prevents false scientific provenance at the serialization boundary.

### F07 — P1: Enforce one execution budget across every public parallel path

**Evidence:** [`parallel.rs`](src/parallel.rs#L39) promises a per-method cap, but universal tensor `zip`, `zip_into`, `dot`, `hermitian_dot`, and `matmul` use unrestricted Rayon iterators. Integrators, thermostat, particle boundary, observers, and mask counting also omit budget helpers. A public custom boundary recorded **four participating workers with `set_max_threads(Some(1))`** when applied to 20,000 particles inside a four-worker pool.

**Change:** capture the operation budget once, apply it consistently to chunked work, and use serial paths for small work or a one-worker budget. Share this internal mechanism across foundational and model operations. Keep pool ownership with the application.

**Acceptance:** verify actual worker participation, not only setter/getter round trips, in caller pools of different sizes. Measure tiny operations and nested application workloads. Check indexed RNG replay separately from floating reductions: schedule-independent randomness does not automatically guarantee bitwise-identical floating aggregates.

### S03 — Indexed randomness and Rayon: batch independent draws inside bounded jobs

[`IndexedRng::indexed_word`](src/rng.rs#L203) mixes independent coordinates using shifts, XOR, and wrapping multiplication. Batch independent item/component coordinates across SIMD lanes and hoist seed/step/domain prefixes that are identical for that batch. Preserve the exact coordinate-to-word function. The indexed uniform/Bernoulli filler loops in [`dense_rand.rs`](src/math/tensor/rank_n/dense_rand.rs#L581) are good initial candidates; split per-element item/component division where a fixed-dimensional loop can avoid it.

`uniform_index` needs full-width products and independent rejection counters. A rejected lane must not advance another item's draw stream. Benchmark this separately from word generation because wide integer multiplication, conversion, and rejection masks have target-dependent costs. [`standard_normal`](src/rng.rs#L195) adds logarithm, cosine, and square root; start with exact batched words and existing transforms. Replacing transforms with vector approximations or reusing the second Box–Muller output can change existing indexed samples and needs an explicit sequence/version decision.

Stateful fillers use fixed RNG lanes and dependency RNG implementations. Inspect their generated code before duplicating vector implementations; preserve lane partitioning and draw consumption. SIMD batching must not become another source of thread-count-dependent sequences.

Apply SIMD **inside each sufficiently large Rayon chunk** under F07's operation budget. A one-thread cap should still permit SIMD. Choose chunks large enough for several SIMD blocks and a safe tail; avoid one Rayon task or repeated CPU detection per vector. Check both per-kernel latency and memory-bandwidth saturation as thread count increases.

## 4. Array interchange

### F08 — P2: Close Python validation gaps before optimizing ingestion

**Evidence:** [`_numeric_values`](python/numpy_support.py#L105) accepts arbitrary objects for `i128`/`u128`, and complex conversion does not recheck finiteness after narrowing. Local probes accepted **`u128: [-1]`, `i128: ["wrong"]`, and `complex_f32: [[1e100, 0]]`**, with the last becoming infinity. [`_lattice_to_ndarray`](python/numpy_support.py#L93) also accepts nonfinite values despite the helper's validation claim.

**Change:** check integer types, booleans, and 128-bit ranges before constructing object arrays; check complex values after dtype conversion. Define lattice validation according to its actual schema, which lacks the tensor scalar discriminator. Use shared Rust-produced valid fixtures and corresponding malformed fixtures for the adapter.

**Ergonomics:** expose a public payload-to-array adapter alongside the path convenience function, allowing callers who already parsed JSON to avoid another file/read boundary. Keep sparse-to-NumPy materialization explicit, with an optional caller-selected allocation limit for large logical shapes.

**Acceptance:** reject the demonstrated invalid inputs and preserve supported dense, sparse, complex, and lattice documents. Do not claim schema parity merely because current tests pass.

### S04 — Serialization boundaries: vectorize validation, preserve the schema

[`ensure_finite`](src/math/io/json.rs) and dense scalar conversion are potential SIMD scans after correctness fixes. Batch finiteness/range classification, then locate an offending scalar within a failing block so diagnostics retain their location and meaning. Sparse canonical-index validation may benefit from contiguous comparisons, but sorting and hash-map traversal are different workloads.

Keep Serde's schema and Python dtype validation unchanged by SIMD. Do not introduce a new JSON parser or duplicate NumPy's bulk operations as the first optimization: profile parsing, validation, and copying separately. Python object arrays for 128-bit integers are not a straightforward packed-numeric SIMD path. Prioritize dense computational kernels before this boundary.

## 5. Spatial primitives and pair generation

### F09 — P1: Validate usable boundary geometry at construction

**Evidence:** [`validate_bounds`](src/space/continuous/boundary.rs#L137) accepts empty bounds. `PeriodicBox::new(&[], &[])` succeeds, then `apply_positions(&mut [])` **panics through division by zero**. Endpoint finiteness checks also do not establish that derived spans, or doubled reflecting spans, are representable.

**Change:** reject zero-dimensional geometry or deliberately implement it consistently; the rest of PiP generally rejects zero dimensions. Validate the arithmetic required by each boundary before accepting its bounds. Report axis and invalid derived extent through a structured error.

**Acceptance:** constructor-to-operation tests cover empty bounds, mismatched rank, extreme finite endpoints, and ordinary wrapping/reflection. Expected input validation failures must not panic.

### F10 — P2: Remove per-position boundary scratch allocation

**Evidence:** [`ReflectBox::apply_position`](src/space/continuous/boundary.rs#L337) creates a flip-mask vector per call. Bulk application to 100,000 three-dimensional positions made **100,000 allocation calls**, even when every position was inside the box. The default position-and-velocity method also allocates one mask per vector.

**Change:** implement position-only reflection without a mask and update position/velocity pairs directly, or reuse scratch per execution chunk for extension implementations. Preserve support for arbitrary dimension. The particle adapter can similarly process position and velocity together instead of allocating a full-system flip mask.

**Acceptance:** bulk allocation count is zero or bounded by execution chunks, not particle count. Check multiple wall crossings and velocity-flip parity.

### F11 — P1: Restore pair-generator allocation stability

**Evidence:** [`PairGenerator::new`](src/space/discrete/square_lattice/pairing.rs#L197) describes an allocation-stable generator. [`refresh_at`](src/space/discrete/square_lattice/pairing.rs#L280) allocates three complete vectors for independent pairs and replaces all caches. A refresh of 100,000 two-dimensional pairs requests **4,800,048 bytes in six allocations** after warm-up. Kernel paths also use the copying vector-list edit helper.

**Change:** after F03, write directly into the generator's owned dense caches and retained workspaces. Keep source/direction/length scratch allocated across sweeps. Treat explicit sweep/domain coordinates as the reproducibility contract.

**Acceptance:** warmed independent, nearest-neighbor, and radial refresh paths have no pair-count-dependent allocations. Repeating a sweep reproduces every coordinate; changing pool size or cap preserves indexed outputs. Check `target = source + displacement` and boundary-aware consumption.

### S05 — Spatial kernels: prioritize lattice interiors, then boundaries and pairing

**Laplacian: high priority.** [`SquareLatticeGeometry::laplacian`](src/space/discrete/square_lattice/representation.rs#L222) already accepts input/output slices, but visits each site through a small component loop and resolves both neighbors per axis through division/modulo and boundary normalization. Split interior spans from boundary faces; use fixed neighbor offsets and contiguous vector loads for the interior, retaining a general edge kernel. Vectorize across sites when `components == 1` and across contiguous site/component blocks otherwise. Hoist inverse spacings and geometry work. Preserve configured boundary semantics, small extents where faces overlap, and per-output arithmetic order.

**Continuous boundaries:** flatten uniform clamp work, or process blocks of 2D/3D positions with repeated axis bounds and masks. Reflection/periodic wrapping require correct floor/remainder behavior, multiple-crossing parity, and finite-value handling; a simple subtract-once approximation is not equivalent. Remove F10's per-position allocation first. Custom `ContinuousBoundary` implementations retain their general path.

**Pair generation and spatial sampling:** [`assemble_nearest_neighbor`](src/space/discrete/square_lattice/pairing.rs#L460) is principally copy/add work and can traverse whole dense chunks. Radial pair assembly and per-axis Gaussian/uniform scaling can batch independent coordinates after F11. Preserve Rust's float-to-integer conversion semantics in radial displacement generation and indexed RNG identities. Variable-rank coordinate conversion, jittered-grid indexing, and `powf` kernel sampling need separate assessment rather than assuming all stages vectorize equally well.

**Acceptance:** compare every output with the general kernel at dimensions 1/2/3 and larger, component counts around SIMD widths, tiny grids, anisotropic spacing, boundary faces/corners, and large overshoots. Benchmark interior-only throughput and the complete operation including boundary and packing costs.

## 6. Particle state and ready-made models

### F12 — P1: Borrow particle columns and fuse repeated passes

**Evidence:** [`ExplicitEuler`](src/models/particles/integrator.rs#L123) copies input columns and uses copying edits for output columns. Shared [`state` helpers](src/models/particles/state.rs#L72) allocate mass/mask vectors. One warmed step at 100,000 particles × 3 dimensions requests **9,800,044 bytes in 12 allocation calls**. Observers copy velocities and inverse masses; energy and temperature requested separately repeat the same gathering and reduction.

**Change:** build short-lived validated borrows over the required columns, then process dense rows in one pass where equations permit it. For explicit Euler, retain each component's old velocity before updating it. Share kinetic aggregates between energy/temperature calculations. Reuse deliberate scratch for sparse or transactional paths. Do not cache validation across publicly mutable attributes without an invalidation mechanism.

**Ergonomics:** retain simple `apply`/`observe` entry points. Any optional workspace should have a straightforward default path and documented reuse behavior.

**Acceptance:** eliminate full-column copies in dense stepping; measure allocations per complete step and per observation interval. Verify both integrators against their equations and test alive/rigid selection, heterogeneous mass, and both backends.

### F13 — P1: Make failed thermostat steps leave state and RNG position consistent

**Evidence:** [`LangevinThermostat::apply`](src/models/particles/thermostat.rs#L183) validates mass and noise amplitude inside a mutating parallel loop. `edit_values` commits its buffer even when that closure returns `Err`. With velocities `[1,1]`, inverse masses `[1,-1]`, temperature 0, friction 1, and `dt=1`, the probe returned an error with velocities **`[0.36787944117144233,1]` and counter 0**. Retrying can apply a second update with the same RNG step to already-modified particles.

**Change:** validate all included-particle preconditions and derived amplitudes before writing, then perform an infallible update and advance the counter. Alternatively commit staged output only on success. Establish the same failure contract for fallible bulk edits and custom boundary adapters.

**Acceptance:** place invalid inputs early and late in a multi-particle state; after failure, both state and counter must match the pre-call snapshot across pool sizes. Successful continuation/replay must remain reproducible.

### F14 — P2: Make the ordinary simulation loop and mass updates harder to misuse

**Evidence:** [`create_template`](src/models/particles/create_state.rs#L126) creates independently mutable `ATTR_M` and `ATTR_M_INV`, while force, thermostat, and observer calculations use inverse mass. Updating only the mass attribute does not update those calculations. Networks add into acceleration, and integrators explicitly leave acceleration uncleared. The sole [example](examples/basic_particle.rs) applies one network without showing a repeated time step.

**Change:** add a validated mass-setting operation that maintains both fields, or select one authoritative representation. Document zero/infinite-mass conventions consistently. Supply a small executable loop showing acceleration reset, force accumulation, integration, boundary/thermostat ordering chosen for that scheme, and periodic observation. Show shared Rayon-pool setup in an application example.

**Acceptance:** a mass change affects forces/observables consistently; a two-step example cannot accidentally retain the previous step's acceleration. Keep scheduling, checkpoint files, and workflow orchestration outside PiP.

### S06 — Particle models: process blocks of particles, respecting masks

After F12, Euler updates become contiguous multiply/add work and are prime SIMD candidates. Specialize common dimensions internally or process flat blocks; preserve explicit Euler's old-velocity dependency. For alive/rigid selection, use blocks of active particles or masks, with an efficient all-active path. Inactive lanes must retain their original values; avoid unsafe out-of-range loads and unintended evaluation of invalid masses in skipped lanes.

The current particle representation is structure-of-arrays **by attribute**, but each position/velocity column contains interleaved vector components. Do not force users to pad 3D vectors or change tensor/Serde layouts for SIMD. If internal deinterleaving or a block layout helps, reuse scratch and include conversion costs in the result; avoid repacking the entire simulation every step.

Kinetic energy/temperature can vectorize squared velocities, inverse-mass scaling, and masks; accumulation still needs S02's numerical policy. Langevin updates can batch the final velocity arithmetic and supported RNG work from S03. F13's complete prevalidation must precede mutation. Shared scalar coefficients such as the thermostat's `exp(-gamma * dt)` are already computed once per step; vectorizing that one call provides no throughput benefit.

**Acceptance:** test both Euler variants, alive/rigid mixtures, nonuniform mass, thermostat continuation, scalar tails, and failed updates. Benchmark complete steps and observation intervals so faster arithmetic is not concealed by retained gathers, reconstruction, or RNG transforms.

## 7. Advanced spatial and interaction engines

### F15 — P1 for validation, P2 for scaling: Bound neighbor-grid construction and simplify safe querying

**Evidence:** [`NeighborList::new`](src/engines/soa/neighbor_list.rs#L97) uses saturated products, allocates a bucket header for every cell, and materializes `3^dimension` neighbor offsets. `ParticleNeighborList::from_box(&[1.0], f64::MIN_POSITIVE)` **panics with capacity overflow**. Rebuild and traversal scan the full cell domain. The [particle wrapper](src/models/particles/interactions/neighbor_list.rs#L107) copies positions on rebuild and again on collection, and only exposes an allocating pair collector. Collection before rebuilding, or after moving particles between cells, can silently miss pairs; the object does not track freshness.

**Change:** check span/count/stride/stencil arithmetic before allocation and return actionable geometry/capacity errors. Benchmark occupied-cell storage or an explicit dense-versus-occupied policy for dilute domains. Add a normal rebuild-and-query operation plus callback/output-buffer collection; document the validity requirements of the separate fast path. Consider displacement-based reuse only after freshness semantics are established.

**Acceptance:** compare pairs against brute force for small configurations, including moved and dead particles. Exercise huge sparse domains without allocating a dense cell grid. Document nonperiodic distance semantics prominently; wrapping positions alone does not implement minimum-image neighbor search.

### F16 — P2: Reduce graph copying and pair-key allocation before parallelizing forces

**Evidence:** network [`insert_many`](src/models/particles/interactions/spring_network.rs#L214) validates a batch, then clones the entire network. [`insert_all_to_all`](src/models/particles/interactions/power_law.rs#L247) adds another clone around that route. [`InteractionNodes`](src/engines/soa/interaction.rs#L59) stores even two-endpoint keys in vectors; lookups construct temporary keys, and insertion duplicates node storage. Both physical network force loops are serial and allocate full-system acceleration scratch. Advanced [`stored_entries`](src/advanced.rs#L112) also returns an allocated vector for every traversal.

**Change:** retain transactional insertion through validated staging and a commit path, avoiding redundant full-graph clones. Evaluate inline pair keys or a pair-specialized internal representation. Add borrowed/callback stored-entry traversal for advanced custom kernels. After F03/F07/F12, benchmark force evaluation and compare deterministic chunk reduction, particle adjacency, or conflict-free edge groups before selecting parallel accumulation.

**Acceptance:** measure repeated small batch edits on a large graph, replacement/lookup allocation, and force cost versus edge count. Preserve duplicate rejection, rollback, pair identity, equal-and-opposite forces, and selection behavior. Avoid unbounded per-worker full-system buffers as a default optimization.

### S07 — Advanced engines: separate vectorizable arithmetic from irregular access

**Neighbor filtering:** batch candidate-pair distance and cutoff calculations in [`ParticleNeighborList::collect_pairs`](src/models/particles/interactions/neighbor_list.rs#L121). The candidates reference scattered rows, so compare direct gathers with reused packed position blocks. Preserve alive checks, unique pair identity, current strict cutoff/coincidence rules, and collection order unless explicitly documented otherwise. SIMD does not remove F15's grid-size and freshness problems.

**Force networks:** batch pair displacements, squared distances, spring coefficients, and force components. Heterogeneous laws/cutoffs and `powf` in power-law forces reduce lane uniformity. Gather/scatter and multiple edges writing one particle are the main constraints: SIMD lanes can conflict even inside one thread. Use conflict-aware batches or staged force contributions followed by controlled accumulation; account for packing and reduction memory. Grouping edges by law may improve execution but must preserve or explicitly specify accumulation order.

**Structured matrices:** [`Diagonal`](src/math/tensor/rank_2/matrix/structured.rs#L17) owns a contiguous diagonal and should multiply vectors in O(n) with a packed elementwise kernel. Symmetric/triangular backends use sparse canonical storage; exploit their support before considering packed tiles. Do not expand a diagonal/triangular matrix to a dense square just to use SIMD.

**Limited candidates:** hash-table topology edits, allocation, individual Fenwick-tree updates/searches in [`DynamicWeightedIndex`](src/sampling/weighted.rs), and the dependent swaps in [`shuffle_slice_indexed`](src/sampling/shuffle.rs#L8) do not naturally become efficient consecutive-lane arithmetic. Independent batches may offer opportunities, but keep their exact checked-weight and permutation semantics. SIMD work should follow evidence of a bottleneck here, after higher-throughput kernels above.

## SIMD implementation and validation gates

1. **Create internal slice kernels first.** Resolve backend and validate dimensions once, then pass borrowed input/output slices to a concrete kernel. Begin with auto-vectorization; use explicit intrinsics where final disassembly and benchmarks show a material gap. Scalar fallback is always available. All existing constructors, mathematical types, and normal operation signatures can remain unchanged.
2. **Use portable runtime dispatch for optional instructions.** Start with generic code plus selected x86/x86-64 AVX2 kernels; evaluate AArch64 NEON on suitable hardware and AVX-512 only where supported and measured. Gate each kernel on its actual required features, including FMA separately where used. Keep architecture-specific code behind target configuration and safe wrappers. Rust supports this through architecture intrinsics and runtime feature detection. [Rust architecture documentation](https://doc.rust-lang.org/stable/core/arch/).
3. **Keep the supported toolchain.** `std::simd` is experimental in the consulted documentation, and a direct probe fails on the installed Rust 1.97.1 with `portable_simd` E0658. Use stable facilities or evaluate a maintained abstraction against PiP's MSRV and dispatch needs; no new dependency is selected by this review. [Rust portable SIMD documentation](https://doc.rust-lang.org/std/simd/index.html).
4. **Retain distributable defaults.** `target-cpu=native` is useful for local comparisons and application builds intended for matching machines. Do not set it globally in PiP or require users to set it to get acceleration. Use unaligned loads unless alignment is established; ordinary slices must work. Handle short inputs and tails without reading outside initialized storage.
5. **Specify numerical guarantees before changing arithmetic.** Independent elementwise operations should preserve existing behavior. Packed floating reductions can change summation order, and vector transcendental functions may need a separate implementation. Keep approximate math/FMA/reordered accumulation from silently changing current results or RNG provenance. SIMD can be introduced incrementally without relaxing these guarantees. [LLVM vectorization guidance](https://llvm.org/docs/Vectorizers.html).
6. **Verify actual generated instructions and end-to-end benefit.** Inspect final linked release binaries with PiP's LTO settings; seeing an XMM/YMM register alone is not proof of packed arithmetic. Test default and host-specialized builds, internally forced dispatch paths, and scalar fallback. Cover odd sizes, vector-width boundaries, unaligned subslices, masks, nonfinite values, signed zero, subnormals, integer extremes, sparse occupancy, and indexed replay. Compare allocations, latency, bandwidth, and whole-step throughput at multiple Rayon caps. Add ARM/AVX-512 claims only after validation on those targets.

Implement S01's semantics and S02's dense kernels first, then advance through S03–S07 alongside their corresponding F findings. No SIMD speedup is claimed from source inspection alone; the compiler probes establish opportunities, not timing results.

## Delivery order and validation gates

Work in the numbered layer order. Within a layer, repair invariants before optimizing; F01's narrow optimization can be deferred after scalar semantics are confirmed. Container reuse and execution-policy fixes are prerequisites for higher-layer tuning. Do not add pools, workflow configuration wrappers, or alternate persistence contracts to solve these costs.

For each layer, land its focused correctness regressions, then measure representative operations. Introduce a dedicated benchmark harness covering logical size, sparse occupancy, matrix aspect ratio, particle/pair count, graph density, and caller-pool size. Record allocation calls/bytes as well as latency; compare first-use and warmed operation. Keep end-to-end stepping benchmarks separate from construction benchmarks.

Every public expensive operation should state time complexity, temporary memory, result backend, output reuse, and error-mutation behavior. Those details are currently uneven despite the README directing users to complexity sections. New examples should be compile-checked and show the basic/model API before advanced escape hatches.

Baseline verification passed: **41 Rust unit/integration tests**, **one doctest**, **six Python tests**, and `cargo fmt --all -- --check`. The disposable probes demonstrated defects not covered by that baseline. Allocation measurements count requested bytes, not peak resident memory. The scalar timing is a local microbenchmark; other proposed speedups remain unmeasured. The separate workflow review has not started.

# Numerical execution and storage costs

PiP selects execution kernels internally. Ordinary `Tensor`, `Matrix`, and
`VectorList` code does not need CPU-specific types, padded dimensions, alignment
annotations, or an application-owned SIMD dispatcher. Backend selection remains
explicit and serialization retains the selected representation.

## SIMD and portability

Dense `f32` and `f64` elementwise add/subtract/multiply/divide and scalar scaling
use runtime CPU detection on x86 and x86-64. Fully active f64 Euler
updates use the same feature dispatch across flat component blocks. Chunks of at least 128 elements may
use AVX-512F; otherwise chunks of at least 32 may use AVX2. These are initial
dispatch thresholds, not a promise that the widest instruction set is fastest
on every processor. Feature detection includes the operating system's support
through Rust's standard detection facilities. Kernels use only the features
declared on their implementation; these floating kernels do not require the
AVX-512 integer extensions or FMA.

Short chunks and other architectures use portable Rust loops. The compiler may
vectorize those loops too. Both explicit paths use unaligned loads/stores and
scalar tails: arbitrary valid slices and dimensions are supported. AVX-512
implementations compile on the supported Rust toolchain; the development host
has AVX2 but no AVX-512, so AVX-512 execution and performance remain unverified
there. Feature-gated regression tests execute those kernels on suitable hosts.

No project-wide `target-cpu=native`, nightly feature, or extra dependency is
required. Applications intentionally building for their own machine can use
native CPU flags, but such executables need matching deployment hardware.

## Numerical behavior

SIMD multiplication and addition remain separate. PiP does not enable fast math,
implicit fused multiply-add, approximate reciprocal/square root, or flushing of
subnormal values. IEEE classifications and signed-zero behavior are tested;
NaN payload bits are not a portable numerical contract. Integer primitive
arithmetic retains its Rust overflow behavior, while scalar magnitude operations
retain their explicit saturation and RNG mixing retains explicit wrapping.

Tensor sums and dot products accumulate in logical row-major order. Sparse sums
sort stored entries before accumulating. Matrix multiplication processes each
output in increasing inner-index order, including when output columns run in
parallel. This avoids changing these operations' arithmetic order with SIMD
width or worker count. Other public reductions document their own ordering;
indexed randomness guarantees do not imply a global bitwise-reproducibility
guarantee for every floating operation.

## Reusing storage

Allocating floating scale initializes its result in one source-to-output pass,
with workers writing disjoint chunks. It performs no full-buffer clone or zero
initialization before scaling.

Dense `fill`, `map_in_place`, elementwise `*_into`, and `matmul_into` retain
the destination allocation. Shape validation happens before mutation. Arithmetic
panics and panicking user map callbacks do not provide rollback. General sparse
maps evaluate all logical coordinates, including zeros, and stage the result.
Vector-list edits internal to model operations borrow dense memory directly;
model-specific fallible operations have their own validation/commit contracts.

`VectorList::vector_into` and `axis_into` copy coordinates into caller-provided
slices without allocating. Vector norms allocate the result and, for sparse
inputs, ordered entry scratch, rather than copying the full logical input.

The caller's active Rayon pool runs sufficiently large disjoint chunks.
PiP captures a per-operation budget and caps those chunks; a one-thread budget
still allows SIMD. Pool ownership and application-level concurrency remain the
application's responsibility. Do not build a pool inside every simulation step.

## Sparse costs

Sparse zero fill clears stored entries while retaining capacity. Zero-preserving
built-in transforms, transpose, and casts visit stored entries. Sparse sums and
norms sort entries to preserve accumulation order. Sparse/sparse addition,
subtraction, and multiplication visit the union of support, including entries
whose partner is implicit zero; that preserves nonfinite products.

Division visits all logical values because `0 / 0` cannot be omitted. Scaling
by a value that turns zero into NaN similarly takes a logical-domain path.
Finite sparse/sparse matrix multiplication joins stored entries in inner-index
order; nonfinite inputs take the general path so implicit-zero products remain
observable. Dense output still requires work proportional to its output size.

Hash lookup and sparse insertion are not contiguous SIMD workloads. PiP does not
materialize a sparse logical domain just to enable SIMD. Always measure result
occupancy as well as input occupancy when choosing a backend.

## Validation

Tests exercise portable short paths and detected SIMD paths, unaligned slices,
vector-width boundaries, nonfinite values, subnormals, and signed zero. A
single-worker allocation regression checks warmed dense output reuse without
mixing allocations from unrelated test threads. Inspect final linked release
assembly when evaluating vectorization: pre-LTO library assembly and vector
register names alone do not establish packed numerical execution.

## Spatial kernels

Pair generators own and reuse their source/displacement/target buffers and
sampling workspaces across explicit sweeps. Built-in reflection and periodic
position/velocity operations allocate no per-position scratch. Empty geometry
and unrepresentable spans fail during construction. Custom boundary callbacks
retain their own allocation and failure behavior.

The lattice Laplacian uses contiguous interior spans with fixed neighbor offsets
and separate boundary spans; inverse spacings are cached in validated geometry.
It allocates no per-call scratch and retains increasing-axis accumulation order.

## Particle operations

Dense Euler integration, Langevin updates, built-in particle boundaries and
kinetic observations borrow validated columns. Euler updates are fused per row;
thermostat amplitudes are checked for every included particle before mutation.
Failures preserve velocities and the thermostat counter. Custom boundaries stage
both position and velocity unless they explicitly promise infallibility after
shape validation; external callback side effects are outside this guarantee.

`kinetic_summary` computes energy, temperature and population together with no
dense input copies. Up to 16,384 particles use one ordered pass. Larger inputs
use fixed 16,384-particle blocks and combine block results in order, with bounded
parallelism and O(particles / 16,384) scratch. The grouping is independent of
thread count but can round differently from a single sequential reduction. Sparse model updates may still
stage logical columns. `set_mass` validates and writes mass and inverse mass
together; rigid flags express fixed bodies. Force routines permit zero inverse
mass, while thermostats and kinetic observers require positive finite inverse
mass for included particles. See `examples/basic_particle.rs` for a complete loop.

## Neighbor and force engines

Neighbor lists store occupied cells, with checked logical counts and strides.
Particle/cell records are sorted in reusable contiguous storage; compact cell
ranges and a hash index replace individual cell vectors. Rebuild costs
O(n log n) worst-case and allocates no particle-dependent storage after capacity
is warmed. Hash iteration never determines pair order. Queries retain only two
dimension-sized coordinate buffers in addition to caller-owned output.
For high ranks, a bounded-stencil policy falls back to comparing occupied cells
instead of allocating exponentially many offsets. That fallback costs
O(occupied_cells² × dimension). Nonfinite rebuild inputs fail before mutation.
Particle queries are nonperiodic and use raw Euclidean distance: wrapping
positions does not provide minimum-image distances. Prefer
`rebuild_and_collect_into` to keep candidates fresh and reuse the pair buffer.
Separate queries require unchanged positions since rebuild; count changes and
queries before the first rebuild are rejected, but arbitrary position edits
cannot be detected automatically.

Graph batch insertion validates incoming entries before committing, without
cloning the existing graph. Pair lookup and replacement use stack lookup keys;
owned public interaction-node types and serialization remain unchanged.
`RawStorage::for_each_stored_entry` provides object-safe allocation-free traversal
for PiP containers, with unspecified sparse order.

Force application validates all masses/endpoints before updating acceleration
and borrows dense inputs. Edges update shared destinations serially in stable
slot order. Contributions now add directly to existing acceleration, which can
round differently from adding a separately summed contribution column. Explicit
conflict-aware parallel force schemes remain a benchmark-led future choice;
per-worker full-system buffers are not the default. Diagonal matrix-vector
products take O(n) for finite inputs; nonfinite input uses the general path to
preserve implicit-zero products.

## Measuring and tuning

Large-data throughput and memory use are the primary targets. See
[BENCHMARKS.md](BENCHMARKS.md) for the opt-in harness, recorded measurements and
allocation comparisons. Elementwise SIMD operates independently of thread count.
Allocating scale and transpose currently wait until 262,144 elements before
splitting work across workers. The scale kernel initializes fresh output in
worker chunks, avoiding the earlier copy-then-distribute cost at large sizes.
Thresholds are implementation choices and remain subject to large-data evidence.

Fully active particle masks are detected inside each Euler worker chunk,
allowing contiguous SIMD across particle boundaries without a serial full-mask
prepass. Chunks containing mixed alive/rigid selections retain the validated row
path. Runtime AVX-512 remains available without changing the
interleaved layout or requiring 3D-vector padding.

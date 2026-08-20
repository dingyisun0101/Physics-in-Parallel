# Physics in Parallel

Physics in Parallel is a Rust crate for building physics-oriented numerical simulations from reusable layers. The design goal is to let domain users work with familiar concepts such as scalars, tensors, spaces, particles, boundaries, laws, and interactions while hiding backend memory layout and parallel traversal details.

The crate is organized from lower-level infrastructure to higher-level model code:

```text
math -> space -> engines -> models
```

- `math` defines scalar algebra, rank-N tensors, matrices, vector batches, random fillers, and math IO.
- `space` defines continuous-space utilities and discrete square-lattice spaces.
- `engines` defines model-agnostic runtime storage, reducers, interaction topology, and neighbor-list infrastructure.
- `models` defines validated physical laws and canonical massive-particle model pieces.

Common import:

```rust
use physics_in_parallel::prelude::*;
```

Narrower imports are also available:

```rust
use physics_in_parallel::math::prelude::*;
use physics_in_parallel::space::prelude::*;
use physics_in_parallel::engines::prelude::*;
use physics_in_parallel::models::prelude::*;
```

## Examples

The runnable examples are documented in [EXAMPLES.md](EXAMPLES.md). That guide
introduces each example, explains what it does, shows the command-line
arguments, and describes how to interpret benchmark or demonstration output.

## Serialization and Persistence Interop

PiP's scientific containers implement Serde directly. They can therefore be
encoded with `serde_json`, embedded in an application's own serializable
records, or passed to any independent storage or workflow library that accepts
`Serialize`. Typed JSON reconstruction uses the corresponding `Deserialize`
implementation:

```rust
use physics_in_parallel::math::{Dense, Tensor};

let tensor = Tensor::<f64, Dense>::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
let json = serde_json::to_string(&tensor)?;
let restored: Tensor<f64, Dense> = serde_json::from_str(&json)?;
assert_eq!(restored.shape(), tensor.shape());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Direct Serde calls are the canonical path. They serialize dense storage from
borrowed slices and stream heterogeneous attribute columns without cloning the
scientific payload first. Encoding must still allocate or write the resulting
JSON bytes. Convenience methods that return a `String`, `serde_json::Value`, or
owned payload representation necessarily allocate and are intended for
inspection and small standalone operations.

The wire format is explicit and self-describing:

- every current document carries `version: 1`;
- homogeneous numeric payloads carry a stable `scalar` identifier;
- dense containers use row-major `shape` and `data`;
- sparse tensors and matrices use sorted flat `indices` and matching `values`;
- `AttrsCore` and `PhysObj` preserve heterogeneous column types and stable
  attribute IDs; and
- malformed shapes, scalar mismatches, non-canonical sparse entries, and
  non-finite JSON numbers are rejected during reconstruction.

PiP deliberately does not own chunking, queues, checkpoint cadence, directory
layouts, or run metadata. Those policies belong to the consuming application
or workflow layer. The dependency direction is one-way: such systems may
depend on PiP's Serde types, while PiP has no dependency on
`scientific-workflow` or any other workflow framework.

## Design Rules

Core consistency rules used across the crate:

- Lower modules should be reused by higher modules. For example, particle boundaries call `space::continuous` boundary logic, particle randomization calls `space::continuous::sampling`, and particle interactions use `engines::soa::Interaction`.
- Add a public type or trait only when no existing abstraction can express the
  behavior. Extend the closest existing concept before creating a parallel API.
- Optional behavior belongs in one configurable interface with documented
  defaults. Every end-user-facing stochastic API accepts only `RngConfig`;
  components reject unsupported methods instead of introducing parallel RNG APIs.
- Public APIs should expose physical or mathematical concepts, not backend layout. Internal storage helpers are kept behind `pub(crate)` where possible.
- Type-preserving math operations keep the same scalar type and backend when the operation is mathematically backend-preserving.
- Explicit conversion APIs have explicit names such as `try_cast_to`, `cast_to`, `to_dense`, `to_sparse`, and `to_ndarray`.
- Boolean particle masks use compact numeric storage internally, while users normally call bool-facing helpers such as `set_alive`, `is_alive`, `set_rigid`, and `is_rigid`.

## Supported public API

This is PiP's exhaustive supported API allowlist. Users may rely on the items
below, their public enum variants, and their documented public methods. The
crate-wide `physics_in_parallel::prelude::*` exports the union of the four
domain preludes plus the RNG types. Compiler-visible implementation paths not
listed here are not compatibility promises.

- Randomness: `IndexedRng`, `RngConfig`, `RngConfigError`, and `RngMethod`.
- Math scalars and conversion: `Complex`, `Scalar`, `ScalarCastError`,
  `ScalarSerde`, and `NdarrayConvert`.
- Tensors and random filling: `Backend`, `DenseBackend`, `SparseBackend`,
  `Tensor`, `TensorError`, `TensorResult`, `TensorTrait`, `RandType`,
  `TensorRandError`, `TensorRandFiller`, `TensorRandElement`, `NUM_RNGS`, and
  the `dense`, `dense_rand`, `sparse`, and `tensor_trait` modules. Their
  low-level surface also includes `dense::Tensor`, `sparse::Tensor`, and the
  `errors` helpers `validate_shape`, `checked_num_elements`,
  `ensure_same_shape`, `ensure_index_rank`, and `ensure_rank`.
- Tensor operation functions: `abs`, `conj`, `cross`, `dot`, `elem_div`,
  `elem_mul`, `hermitian_dot`, `hermitian_transpose`, `map`, `matmul`, `norm`,
  `norm_sqr`, `norm_sqr_real`, `scalar_mul`, `size`, `sqrt`, `transpose`,
  `wedge`, `zip_with`, and `assert_same_shape` in `math::tensor::rank_n::ops`.
- Rank-two math: `Matrix`, `DenseMatrix`, `SparseMatrix`, `MatrixBackend`,
  `MatrixError`, `DiagonalMatrix`, `SymmetricMatrix`,
  `AntiSymmetricMatrix`, `UpperTriangularMatrix`,
  `StrictUpperTriangularMatrix`, `LowerTriangularMatrix`,
  `StrictLowerTriangularMatrix`, `RankNDense`, `RankNSparse`, `Diagonal`,
  `Symmetric`, `AntiSymmetric`, `Triangular`, `VectorList`, `DynVectorList`,
  `VectorListRand`, `HaarVectors`, and `NNVectors`.
- Math interchange schema: `JSON_SCHEMA_VERSION`, `FlatPayload`,
  `SparsePayload`, `SparsePayloadParts`, `ToJsonPayload`, `FromJsonPayload`,
  `TensorStringConvert`, and the JSON helpers `checked_num_elements` and
  `scalar_kind`.
- Continuous space: `BoundaryError`, `ClampBox`, `ContinuousBoundary`,
  `PeriodicBox`, `ReflectBox`, `VectorSamplingError`,
  `VectorSamplingMethod`, and `sample_vectors`.
- General and lattice space: `Space`, `BoundaryCondition`, `Kernel`,
  `KernelError`, `KernelType`, `NearestNeighborKernel`,
  `PairGenerationError`, `PairGenerator`, `PairGeneratorConfig`, `PairingMethod`,
  `PowerLawKernel`, `RandPairGenerator`, `SourceMode`,
  `SquareLattice`, `SquareLatticeConfig`, `SquareLatticeConfigError`,
  `SquareLatticeInitMethod`, `UniformDistanceKernel`, `create_kernel`,
  `try_create_kernel`, and `save_square_lattice`.
- Generic engines: `MeanReducer`, `Reducer`, `AttrId`, `AttrsCore`,
  `AttrsError`, `AttrsMeta`, `PhysObj`, `ObjId`, `InteractionId`,
  `Interaction`, `InteractionError`, `InteractionNodes`, `InteractionOrder`,
  `InteractionTopology`, `NeighborList`, and `NeighborListError`.
- Physical laws: `PowerLawDecay`, `PowerLawError`, `PowerLawRange`, `Spring`,
  `SpringCutoff`, and `SpringLawError`.
- Particle attributes: `ParticleSelection`, `ALIVE_FALSE`, `ALIVE_TRUE`,
  `RIGID_FALSE`, `RIGID_TRUE`, `ATTR_A`, `ATTR_ALIVE`, `ATTR_M`,
  `ATTR_M_INV`, `ATTR_R`, `ATTR_RIGID`, `ATTR_V`, `alive_value`,
  `is_alive`, `is_alive_value`, `is_rigid`, `is_rigid_value`,
  `rigid_value`, `set_alive`, and `set_rigid`.
- Particle construction and dynamics: `MassiveParticlesError`,
  `VelocitySamplingMethod`, `create_template`, `randomize_r`, `randomize_v`,
  `ParticleBoundary`, `ParticleBoundaryError`, `ExplicitEuler`, `Integrator`,
  `IntegratorError`, `SemiImplicitEuler`, `LangevinThermostat`, `Thermostat`,
  and `ThermostatError`.
- Particle observation and interactions: `KineticEnergyObserver`,
  `ObserveError`, `Observer`, `TemperatureObserver`, `ParticleNeighborList`,
  `ParticleNeighborListError`, `PowerLawNetwork`, `PowerLawNetworkError`,
  `SpringNetwork`, and `SpringNetworkError`.

The Math, Space, Engines, and Models sections below exhaustively catalogue the
supported constructors and operations for these items. Generated crate
documentation is the exact signature reference.

## Unified RNG Configuration

`RngConfig` is the only randomness input used by PiP's public stochastic APIs.
Its seed and method are independently optional. Missing values select the
receiving component's documented defaults; a missing seed is resolved from
host entropy.

```rust
use physics_in_parallel::prelude::*;

let rng = RngConfig::new(
    Some(42),
    Some(RngMethod::SmallRng),
);
```

Stateful tensor sampling supports `SmallRng`, PCG, and ChaCha methods.
Explicit-step tensor filling and lattice pair generation use
`IndexedSplitMix64`; their output is independent of worker scheduling. Tensor
random fills use the crate-wide static `NUM_RNGS` partition count, currently
32, rather than exposing execution tuning through scientific configuration.
`RngConfig::default()` lets each component select all defaults.

Construction resolves all missing values. Long-lived stochastic objects expose
the resulting `rng_config()`; one-shot sampling functions return it. Use
`RngMethod::name()`, `version()`, and `seed_encoding()` together with
`RngConfig::encode_seed()` when writing workflow provenance.

## Math

Purpose:

`math` is the numeric foundation. It provides scalar traits, generic rank-N tensors, rank-2 matrix wrappers, vector-list batches, random fillers, and IO conversion traits.

### Scalar

Purpose:

`Scalar` gives integers, real floats, and complex values one common number-like interface. It separates operations that preserve the scalar type from projection/construction/casting operations that intentionally cross type boundaries.

Core API:

```rust
T::zero()
T::one()
T::from_real(...)
T::real()
T::imag()
T::conj()
T::abs()
T::sqrt()
T::exp()
T::ln()
T::sin()
T::cos()
T::pow()
T::norm_sqr()
T::norm_sqr_real()
T::try_cast_to::<U>()
T::cast_to::<U>()
```

Core types:

- `Scalar`
- `ScalarSerde`
- `ScalarCastError`
- `Complex`

### Rank-N Tensor

Purpose:

`Tensor<T, Backend>` is the general tensor facade. It lets users address tensors by shape and coordinate while dense/sparse storage and parallel element traversal stay behind the API.

Core creation API:

```rust
Tensor::<T, DenseBackend>::empty(shape)
Tensor::<T, DenseBackend>::zeros(shape)
Tensor::<T, DenseBackend>::from_vec(shape, data)
Tensor::<T, DenseBackend>::from_fn(shape, f)
Tensor::<T, SparseBackend>::empty(shape)
Tensor::<T, SparseBackend>::from_triplets(shape, triplets)
```

Core access API:

```rust
tensor.shape()
tensor.rank()
tensor.size()
tensor.get(coord)
tensor.get_mut(coord)
tensor.set(coord, value)
tensor.fill(value)
tensor.print()
```

Type-preserving math API:

```rust
&tensor + &rhs
&tensor - &rhs
tensor.elem_mul(&rhs)
tensor.elem_div(&rhs)
tensor.scalar_mul(scalar)
tensor.conj()
tensor.abs()
tensor.sqrt()
tensor.transpose()
tensor.hermitian_transpose()
```

Explicit conversion and linear algebra API:

```rust
tensor.try_cast_to::<U>()
tensor.cast_to::<U>()
tensor.to_dense()
tensor.to_sparse()
tensor.dot(&rhs)
tensor.hermitian_dot(&rhs)
tensor.matmul(&rhs)
tensor.cross(&rhs)
tensor.wedge(&rhs)
tensor.norm_sqr()
tensor.norm_sqr_real()
```

Core types:

- `Tensor<T, DenseBackend>`
- `Tensor<T, SparseBackend>`
- `TensorTrait`
- `TensorError`
- `TensorResult`

### Tensor Random Fillers

Purpose:

`TensorRandFiller` fills dense tensor storage in parallel. It is the shared random infrastructure used directly by tensors and indirectly by vector-list random generators and continuous-space sampling.
Indexed fillers also provide exact schedule-independent site/index sampling over
`0..upper` through `try_fill_indices_at`; this is the native backend used by
independent-uniform lattice pairing.

Core API:

```rust
TensorRandFiller::new(rand_type, rng_config)
TensorRandFiller::try_new(rand_type, rng_config)?
indexed_filler.try_fill_indices_at(indices, upper, step, domain)?
filler.refresh(tensor)
filler.try_refresh(tensor)
filler.try_fill_slice(values)
filler.rng_config()
```

Use `RngConfig::default()` for component defaults. The resolved seed, method,
and stream count remain available through `rng_config()` for provenance.

Core types:

- `TensorRandFiller`
- `RandType`
- `RngConfig`
- `RngMethod`
- `TensorRandError`

### Matrix

Purpose:

`Matrix<T, Backend>` is a rank-2 wrapper over rank-N tensor infrastructure. Dense and sparse matrices use rank-N tensor backends; structured matrices such as diagonal, symmetric, antisymmetric, and triangular matrices store only the independent entries and infer the rest.

Core creation API:

```rust
DenseMatrix::<T>::empty(rows, cols)
DenseMatrix::<T>::from_vec(rows, cols, data)
DenseMatrix::<T>::try_from_vec(rows, cols, data)?
SparseMatrix::<T>::empty(rows, cols)
SparseMatrix::<T>::from_triplets(rows, cols, triplets)
DiagonalMatrix::<T>::empty(n, n)
SymmetricMatrix::<T>::empty(n, n)
AntiSymmetricMatrix::<T>::empty(n, n)
UpperTriangularMatrix::<T>::empty(n, n)
LowerTriangularMatrix::<T>::empty(n, n)
StrictUpperTriangularMatrix::<T>::empty(n, n)
StrictLowerTriangularMatrix::<T>::empty(n, n)
```

Core API:

```rust
matrix.shape()
matrix.rows()
matrix.cols()
matrix.size()
matrix.get(i, j)
matrix.set(i, j, value)
matrix.fill(value)
matrix.abs() // dense matrices
matrix.max_abs_real()
matrix.print()
matrix.add(&rhs)
matrix.sub(&rhs)
matrix.elem_mul(&rhs)
matrix.elem_div(&rhs)
matrix.scalar_mul(scalar)
matrix.transpose()
matrix.hermitian_transpose()
matrix.trace()
matrix.matmul(&rhs)
matrix.mul_vector_into(input, output)?
matrix.try_cast_to::<U>()
matrix.cast_to::<U>()
matrix.to_dense()
matrix.to_sparse()
matrix.to_dense_matrix()
```

Core types:

- `DenseMatrix<T>`
- `SparseMatrix<T>`
- `DiagonalMatrix<T>`
- `SymmetricMatrix<T>`
- `AntiSymmetricMatrix<T>`
- `UpperTriangularMatrix<T>`
- `LowerTriangularMatrix<T>`
- `StrictUpperTriangularMatrix<T>`
- `StrictLowerTriangularMatrix<T>`
- `MatrixError`

### VectorList

Purpose:

`VectorList<T>` stores many fixed-length vectors as dense rank-N storage with logical shape `[num_vectors, dim]`. It is used when the natural unit of manipulation is a vector row, not an individual scalar.

Core API:

```rust
VectorList::<T>::empty(dim, num_vectors)
VectorList::<T>::zeros(dim, num_vectors)
VectorList::<T>::from_vec(dim, num_vectors, data)
VectorList::<T>::from_fn(dim, num_vectors, f)
vectors.dim()
vectors.num_vectors()
vectors.shape()
vectors.get(i, axis)
vectors.set(i, axis, value)
vectors.get_vec(i)
vectors.get_vec_mut(i)
vectors.get_vec_owned(i)
vectors.set_vec(i, values)
vectors.axis(axis)
vectors.fill(value)
vectors.print()
vectors.par_for_each_vec(...)
vectors.par_for_each_vec_mut(...)
vectors.scale_vectors_by_list(scales)
vectors.normalize()
vectors.norms_real()
vectors.try_cast_to::<U>()
vectors.cast_to::<U>()
```

Random vector batches:

```rust
HaarVectors::new(dim, num_vectors, rng_config)?
haar.refresh()
haar.rng_config()
NNVectors::new(dim, num_vectors, rng_config)?
nn.refresh()
nn.rng_config()
```

Core types:

- `VectorList<T>`
- `VectorListRand`
- `HaarVectors`
- `NNVectors`

### Math IO

Purpose:

`math::io` handles external-format interop for math containers. Every current
JSON payload carries `version: 1`. Dense containers borrow their row-major data
directly during Serde encoding:

```json
{"kind":"tensor","version":1,"scalar":"f64","shape":[2,2],"data":[1.0,2.0,3.0,4.0]}
```

Sparse tensors and matrices encode only explicit nonzeros through strictly
increasing row-major flat indices:

```json
{"kind":"tensor_sparse","version":1,"scalar":"f64","shape":[1000,1000],"indices":[12,9004],"values":[2.5,-1.0]}
```

Sparse encoding is proportional to `nnz`, never materializes implicit zeros,
and is deterministic regardless of hash-map order. Decoding validates version,
shape, lengths, bounds, ordering, duplicates, and explicit zeros before direct
sparse construction. PiP rejects non-finite numeric values because ordinary
JSON numbers cannot represent NaN or infinity faithfully. Finite `f32` and
`f64` values use exact round-trip parsing.

Prefer `serde_json::to_writer` when writing to a file or buffered stream, and
`serde_json::to_string` only when an owned JSON string is actually needed.
`ToJsonPayload` and `FromJsonPayload` expose owned schema representations for
specialized manipulation; they are not required for ordinary serialization.

Core API:

```rust
serde_json::to_string_pretty(&value)
serde_json::from_str::<T>(json)
value.to_ndarray()
T::from_ndarray(&array)
value.to_tensor_string()
T::from_tensor_string(input)
```

Core types:

- `NdarrayConvert`
- `ToJsonPayload`
- `FromJsonPayload`
- `FlatPayload<T>`
- `SparsePayload<T>`
- `TensorStringConvert`

## Space

Purpose:

`space` adds physical coordinate semantics on top of math data structures. It contains continuous-space tools and discrete square-lattice spaces.

### Continuous Boundary

Purpose:

Continuous boundaries define how real-valued coordinate vectors are returned to an axis-aligned domain. The pure boundary code is independent of particles and can operate on one vector or a flat list of vectors.

Core API:

```rust
PeriodicBox::new(min, max)
ClampBox::new(min, max)
ReflectBox::new(min, max)
boundary.dim()
boundary.min()
boundary.max()
boundary.apply_position(r)
boundary.apply_position_velocity(r, v)
boundary.apply_positions(flat_positions)
boundary.apply_positions_velocities(flat_positions, flat_velocities)
```

Core types:

- `ContinuousBoundary`
- `PeriodicBox`
- `ClampBox`
- `ReflectBox`
- `BoundaryError`

### Continuous Sampling

Purpose:

Continuous sampling fills `VectorList<f64>` values with common coordinate or velocity initialization patterns. Particle state construction delegates to this module for generic continuous-vector randomization.

Core API:

```rust
let resolved_rng = sample_vectors(vectors, method, rng_config)?;
```

Core types:

- `VectorSamplingMethod`
- `VectorSamplingError`

### Space Trait

Purpose:

`Space<T>` gives higher-level code a common interface for spatial containers without exposing each container's storage or boundary implementation.

Core API:

```rust
space.data()
space.dims()
space.linear_size()
space.get(coord)
space.get_mut(coord)
space.set(coord, value)
space.set_all(value)
space.save(path, target_side_length)
```

### SquareLattice

Purpose:

`SquareLattice<T>` represents square, cubic, or hypercubic lattice sites over a tensor-style shape such as `[128]`, `[64, 64]`, or `[32, 64, 16]`.

Core API:

```rust
SquareLatticeConfig::try_new(shape, boundary, spacing)?
SquareLatticeConfig::new(shape, boundary, spacing)
SquareLatticeConfig::periodic(shape)
SquareLatticeConfig::reflective(shape)
cfg.shape()
cfg.boundary()
cfg.rank()
cfg.num_sites()
cfg.tensor_shape()
cfg.spacing()
cfg.strides()
cfg.coordinate(flat)
cfg.neighbor(flat, axis, offset)
cfg.laplacian(input, components, output)?
SquareLattice::<T>::new(cfg, init_method)?
lattice.config()
lattice.initialization_rng_config()
lattice.data()
lattice.downsample(target_shape)
lattice.rescale(target_shape)
```

Core types:

- `SquareLattice<T>`
- `SquareLatticeConfig`
- `SquareLatticeConfigError`
- `SquareLatticeInitMethod<T>`
- `BoundaryCondition`

`SquareLatticeConfig` is the complete serializable spatial configuration. Its
JSON form is `{"shape":[64,64],"boundary":"periodic","spacing":[1.0,1.0]}`.
Deserialization and
`try_new` reject empty shapes, zero-length axes, unknown fields, and site-count
overflow, so downstream crates do not need mirror shape, boundary, spacing,
stride, coordinate, neighbor, or finite-difference layout types. `Neumann`
provides zero-normal-gradient edge handling. PiP does not assign a universal
vacancy sentinel: zero remains a valid scientific value unless a model says otherwise.

### Square-Lattice Kernels And Pair Generation

Purpose:

Pairing methods define complete source-target selection rules. Independent
uniform pairing samples both sites directly, independently, and with
replacement; self-pairs are therefore ordinary outcomes with probability
`1 / num_sites`. Kernel pairing instead samples a source and a displacement.
Independent uniform pairing is not represented as a kernel.

`PairGenerator` creates source coordinates, raw displacements, and raw targets
from one resolved `RngConfig` and scientific sweep. All methods produce
`[num_pairs, rank]` buffers with `target = source + displacement`. Sources are
canonical coordinates. Targets are raw coordinates and may be outside the
lattice for displacement kernels; pass them to `SquareLattice` access methods
so the space applies its boundary condition. Do not use raw targets as unchecked
flat storage indices.

Every random value is indexed by seed, sweep, domain, pair, and component, so
generated batches are identical across Rayon worker counts. Independent uniform
pairing uses exact integer site sampling from PiP's `TensorRandFiller`; it does
not use floating-point coordinate scaling or downstream randomness.

Core API:

```rust
try_create_kernel(kernel_type)
kernel.sample_indexed(rng_config, sweep, sample_index)?
kernel.sample_batch_indexed(rng_config, sweep, n)?
kernel.kind()
PairGeneratorConfig::independent_uniform(num_pairs, rng_config)
PairGeneratorConfig::kernel(kernel_type, num_pairs, source_mode, rng_config)
PairGenerator::new(shape, config)?
gen.refresh_at(sweep)
gen.method()
gen.rng_config()
gen.sources()
gen.displacements()
gen.targets()
gen.source(i)
gen.displacement(i)
gen.target(i)
```

Core types:

- `Kernel`
- `KernelType`
- `NearestNeighborKernel`
- `UniformDistanceKernel`
- `PowerLawKernel`
- `PairGenerator`
- `PairGeneratorConfig`
- `PairingMethod`
- `RandPairGenerator`
- `RngConfig`
- `RngMethod`
- `PairGenerationError`
- `SourceMode`

`RandPairGenerator` is the deprecated compatibility wrapper for the original
kernel-only constructor. New code should use `PairGenerator`.

Indexed algorithms resolve to `RngMethod::IndexedSplitMix64`. Record the
resolved configuration plus the method's name, version, and seed encoding. PiP
owns the random mapping; a workflow or application remains responsible for
persisting these facts with the simulation record.

### Space IO

Purpose:

`space::io` contains IO behavior for space types. Current ready support is square-lattice JSON and ndarray conversion.

Core API:

```rust
save_square_lattice(&lattice, target_shape, path)
serde_json::to_string_pretty(&lattice)
serde_json::from_str::<SquareLattice<T>>(json)
SquareLattice::from_ndarray(&array, boundary)
lattice.to_ndarray()
lattice.serialize()
```

## Engines

Purpose:

`engines` provides model-agnostic runtime infrastructure. The ready backend is structure-of-arrays storage, where each attribute is stored as a typed vector-list column.

### Reducers

Purpose:

Reducers combine batches of observed values without knowing which model produced the values.

Core API:

```rust
MeanReducer.reduce(values)
```

Core types:

- `Reducer<T>`
- `MeanReducer`

### PhysObj And Attribute Storage

Purpose:

`PhysObj` stores many simulation objects as named typed attribute columns. Each column is a `VectorList<T>` with shape `[n_objects, dim]`. Attribute labels are the normal user path; generated attribute IDs are available for repeated expert lookups.

`AttrsCore` and `PhysObj` implement direct streaming Serde serialization and
typed deserialization. The versioned representation stores each column's stable
attribute ID and scalar kind, so mixed built-in PiP scalar columns reconstruct
without an intermediate `serde_json::Value` tree. Consequently, `PhysObj` can
be used as a typed payload by any independent application or storage framework
that accepts Serde values.

Core API:

```rust
AttrsMeta::empty()
AttrsMeta::new(id, label, comment)
AttrsCore::empty()
core.allocate::<T>(label, dim, n_objects)
core.insert(label, vector_list)
core.remove(label)
core.rename(from, to)
core.contains(label)
core.labels()
core.n_objects()
core.id_of(label)
core.label_of(id)
core.get::<T>(label)
core.get_mut::<T>(label)
core.get_by_id::<T>(id)
core.get_by_id_mut::<T>(id)
core.vector_of::<T>(label, obj)
core.vector_of_mut::<T>(label, obj)
core.set_vector_of::<T>(label, obj, values)
core.dim_of(label)
core.type_name_of(label)
PhysObj::empty()
PhysObj::new(meta, core)
obj.serialize()
obj.save_to_json(output_dir, filename)
```

Core types:

- `AttrsMeta`
- `AttrsCore`
- `AttrsError`
- `AttrId`
- `PhysObj`

### Interaction Storage

Purpose:

`InteractionTopology` maps participating object IDs to stable interaction IDs. `Interaction<T>` combines topology with payload storage, so topology edits and payload edits stay synchronized.

Core API:

```rust
InteractionTopology::new(n_objects)
InteractionTopology::with_order(n_objects, order)
topology.set_order(order)
topology.set_n_objects(n_objects)
topology.prune_n_objects(n_objects)
topology.add(nodes)
topology.remove(nodes)
topology.id_of(nodes)
topology.nodes_of(id)
topology.add_pair(i, j)
topology.remove_pair(i, j)
Interaction::<T>::new(n_objects, order)
Interaction::<T>::with_topology(topology)
interaction.set(nodes, payload)
interaction.get(nodes)
interaction.get_mut(nodes)
interaction.remove(nodes)
interaction.set_pair(i, j, payload)
interaction.get_pair(i, j)
interaction.remove_pair(i, j)
interaction.par_for_each(...)
interaction.par_for_each_payload_mut(...)
```

Core types:

- `InteractionTopology`
- `Interaction<T>`
- `InteractionNodes`
- `InteractionOrder`
- `InteractionError`
- `ObjId`
- `InteractionId`

### NeighborList

Purpose:

`NeighborList` is a cell-linked candidate-pair generator. It emits unique unordered candidate pairs from same/adjacent cells but does not apply a final physical cutoff distance.

Core API:

```rust
NeighborList::new(min, max, cell_width)
neighbor_list.rebuild(positions, n_objects)
neighbor_list.for_each_pair_candidate(|i, j| { ... })
neighbor_list.collect_pair_candidates()
neighbor_list.clear()
neighbor_list.dim()
neighbor_list.num_objects()
neighbor_list.cells_per_axis()
```

Core types:

- `NeighborList`
- `NeighborListError`

## Models

Purpose:

`models` contains concrete physical model pieces built on the lower layers. Current ready modules cover validated law payloads and canonical massive-particle simulation components.

### Laws

Purpose:

`models::laws` stores small validated parameter payloads. These payloads do not know how objects are stored; model adapters decide how to apply them to particle state, lattice sites, or future model objects.

Core API:

```rust
Spring::new(k, l_0, cutoff)
spring.validate()
PowerLawDecay::new(k, alpha, range)
power_law.validate()
```

Core types:

- `Spring`
- `SpringCutoff`
- `SpringLawError`
- `PowerLawDecay`
- `PowerLawRange`
- `PowerLawError`

### Particle Attributes And State Construction

Purpose:

Particle modules use a canonical `PhysObj` layout. Vector attributes `r`, `v`, and `a` have shape `[num_particles, dim]`. Scalar attributes `m`, `m_inv`, `alive`, and `rigid` have shape `[num_particles, 1]`.

Core attribute API:

```rust
set_alive(objects, i, alive)
is_alive(objects, i)
set_rigid(objects, i, rigid)
is_rigid(objects, i)
alive_value(alive)
rigid_value(rigid)
ParticleSelection::AliveOnly
ParticleSelection::All
```

Core construction/randomization API:

```rust
create_template(dim, num_particles)
let resolved_rng = randomize_r(objects, method, rng_config)?;
let resolved_rng = randomize_v(objects, velocity_method, rng_config)?;
```

Core types:

- `ParticleSelection`
- `MassiveParticlesError`
- `VelocitySamplingMethod`

### Particle Boundary

Purpose:

Particle boundary adapters apply `space::continuous` boundary objects to canonical particle state. The continuous boundary owns the geometric rule; the particle adapter owns traversal over `ATTR_R`, velocity updates in `ATTR_V`, and alive/rigid mask handling.

Core API:

```rust
PeriodicBox::new(min, max)?.apply_to_particles(objects)
ClampBox::new(min, max)?.apply_to_particles(objects)
ReflectBox::new(min, max)?.apply_to_particles(objects)
```

Core types:

- `ParticleBoundary`
- `ParticleBoundaryError`

### Particle Integrators

Purpose:

Integrators advance canonical particle `r` and `v` from `a`. They skip dead particles and rigid particles; they do not clear acceleration after stepping.

Core API:

```rust
ExplicitEuler.apply(objects, dt)
SemiImplicitEuler.apply(objects, dt)
```

Core types:

- `Integrator`
- `ExplicitEuler`
- `SemiImplicitEuler`
- `IntegratorError`

### Particle Thermostat

Purpose:

`LangevinThermostat` applies an exact Ornstein-Uhlenbeck velocity update to canonical particle velocities. It honors `ParticleSelection` for alive/dead behavior and always skips rigid particles.

Core API:

```rust
LangevinThermostat::new(tau_target, gamma, rng_config, selection)?
thermostat.apply(objects, dt)
thermostat.tau_target()
thermostat.gamma()
thermostat.rng_config()
thermostat.step_counter()
thermostat.selection()
```

Core types:

- `Thermostat`
- `LangevinThermostat`
- `ThermostatError`

### Particle Observers

Purpose:

Particle observers compute read-only scalar summaries from canonical particle state. `AliveOnly` skips dead particles, while `All` intentionally includes every allocated slot for diagnostics.

Core API:

```rust
KineticEnergyObserver::default().observe(objects)
KineticEnergyObserver::new(selection).observe(objects)
TemperatureObserver::default().observe(objects)
TemperatureObserver::new(selection).observe(objects)
```

Core types:

- `Observer`
- `KineticEnergyObserver`
- `TemperatureObserver`
- `ObserveError`

### Particle Interactions

Purpose:

Particle interaction modules wrap engine-level interaction storage with particle-specific validation and application rules.

Core API:

```rust
ParticleNeighborList::from_bounds(min, max, cutoff)
ParticleNeighborList::from_box(dimensions, cutoff)
neighbor_list.rebuild(objects)
neighbor_list.collect_pairs(objects, selection)

SpringNetwork::empty()
springs.add_spring(pair, k, l_0, cutoff)
springs.add_spring_payload(pair, spring)
springs.get_spring(pair)
springs.remove_spring(pair)
springs.apply_hooke_acceleration(objects, selection)

PowerLawNetwork::empty()
network.add_power_law(pair, k, alpha, range)
network.add_payload(pair, payload)
network.get_power_law(pair)
network.remove_power_law(pair)
```

Core types:

- `ParticleNeighborList`
- `ParticleNeighborListError`
- `SpringNetwork`
- `SpringNetworkError`
- `PowerLawNetwork`
- `PowerLawNetworkError`

## Examples

Runnable examples and benchmark usage are documented in [EXAMPLES.md](EXAMPLES.md).
Common entry points:

```bash
cargo run --release --example spring_network_benchmark
cargo run --release --example power_law_network_benchmark
cargo run --example serde_flat_json
cargo run --example vector_list_ndarray
cargo run --release --example tensor_rand_large_benchmark
cargo run --release --example vector_list_haar_benchmark
```

## Verification

Standard checks:

```bash
cargo fmt --check
cargo test
cargo doc --no-deps
```

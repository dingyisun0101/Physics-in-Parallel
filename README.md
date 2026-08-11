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
  defaults. In particular, random construction accepts an optional seed and
  optional algorithm parameters instead of adding one constructor per combination.
- Public APIs should expose physical or mathematical concepts, not backend layout. Internal storage helpers are kept behind `pub(crate)` where possible.
- Type-preserving math operations keep the same scalar type and backend when the operation is mathematically backend-preserving.
- Explicit conversion APIs have explicit names such as `try_cast_to`, `cast_to`, `to_dense`, `to_sparse`, and `to_ndarray`.
- Boolean particle masks use compact numeric storage internally, while users normally call bool-facing helpers such as `set_alive`, `is_alive`, `set_rigid`, and `is_rigid`.

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

Core API:

```rust
TensorRandFiller::new(rand_type, seed, rng_kind, num_rngs)
TensorRandFiller::try_new(rand_type, seed, rng_kind, num_rngs)?
filler.refresh(tensor)
filler.try_refresh(tensor)
filler.seed()
filler.rng_kind()
```

Every optional argument may be `None`. A missing seed uses host entropy but the
resolved value remains available through `seed()` for provenance records.

Core types:

- `TensorRandFiller`
- `RandType`
- `RngKind`
- `TensorRandError`

### Matrix

Purpose:

`Matrix<T, Backend>` is a rank-2 wrapper over rank-N tensor infrastructure. Dense and sparse matrices use rank-N tensor backends; structured matrices such as diagonal, symmetric, antisymmetric, and triangular matrices store only the independent entries and infer the rest.

Core creation API:

```rust
DenseMatrix::<T>::empty(rows, cols)
DenseMatrix::<T>::from_vec(rows, cols, data)
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
HaarVectors::new(dim, num_vectors, num_rngs)
haar.refresh()
NNVectors::new(dim, num_vectors, num_rngs)
nn.refresh()
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
sample_vectors(vectors, VectorSamplingMethod::Uniform { low, high })
sample_vectors(vectors, VectorSamplingMethod::UniformCentered { box_size })
sample_vectors(vectors, VectorSamplingMethod::GaussianPerAxis { mean, std })
sample_vectors(vectors, VectorSamplingMethod::JitteredLattice { spacings, sigmas })
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

Kernels define reproducible displacement rules for square-lattice workflows.
`RandPairGenerator` creates source coordinates, raw displacements, and raw
targets from one resolved `RandomKey` and scientific sweep. Every value is
indexed by key, sweep, domain, pair, and component, so generated batches are
identical across Rayon worker counts. Boundary interpretation is intentionally
left to `SquareLattice` access methods.

Core API:

```rust
try_create_kernel(kernel_type)
kernel.sample_indexed(key, sweep, sample_index)
kernel.sample_batch_indexed(key, sweep, n)
kernel.kind()
let key = RandomKey::new(optional_seed);
RandPairGenerator::new(shape, kernel_type, num_pairs, source_mode, key)?
gen.refresh_at(sweep)
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
- `UniformKernel`
- `PowerLawKernel`
- `RandPairGenerator`
- `RandomKey`
- `PairGenerationError`
- `SourceMode`

For provenance, record `INDEXED_RANDOM_METHOD`, `INDEXED_RANDOM_VERSION`,
`INDEXED_RANDOM_KEY_ENCODING`, and `RandomKey::encode()`. PiP owns the random
mapping; a workflow or application remains responsible for persisting these
facts with the simulation record.

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
randomize_r(objects, method)
randomize_v(objects, VelocitySamplingMethod::Uniform { low, high })
randomize_v(objects, VelocitySamplingMethod::GaussianPerAxis { mean, std })
randomize_v(objects, VelocitySamplingMethod::MaxwellBoltzmann { tau })
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
LangevinThermostat::new(tau_target, gamma, seed, selection)
thermostat.apply(objects, dt)
thermostat.tau_target()
thermostat.gamma()
thermostat.seed()
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

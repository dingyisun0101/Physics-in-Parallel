# Physics in Parallel

Physics in Parallel is a Rust crate for numerical and physics simulation infrastructure.

The crate is organized as layered modules:

```text
math -> space -> engines -> models
```

Current documentation focus:

- `math`: scalar, tensor, matrix, vector-list, random generation, and math IO.
- `space`: square-lattice spaces, coordinate-pair generation, square-lattice kernels, and space IO.
- `engines`: structure-of-arrays object storage, interaction topology, and neighbor-list infrastructure.

The easiest import for common usage is:

```rust
use physics_in_parallel::prelude::*;
```

For narrower imports:

```rust
use physics_in_parallel::math::prelude::*;
use physics_in_parallel::space::prelude::*;
```

## Module Layout

The refactored core modules use folder-based Rust module layout:

```text
src/math/
    mod.rs
    scalar/
    tensor/
    io/

src/space/
    mod.rs
    discrete/
        square_lattice/
            representation.rs
            displacement.rs
            kernel.rs
    io/

src/engines/
    mod.rs
    soa/
        mod.rs
        phys_obj.rs
        interaction.rs
        neighbor_list.rs
```

## Math

Purpose:

`math` provides the numeric foundation used by the rest of the crate. It defines scalar behavior, generic rank-N tensors, matrix abstractions, vector-list batches, random fillers, and IO conversion utilities.

Common import:

```rust
use physics_in_parallel::math::prelude::*;
```

### Scalar

Purpose:

`Scalar` gives integers, real floats, and complex values one common mathematical interface. It separates type-preserving operations from operations that explicitly change type or project values.

Core structs and traits:

- `Scalar`: common scalar math trait.
- `ScalarSerde`: scalar values that can also be serialized/deserialized.
- `ScalarCastError`: error type for failed explicit scalar casts.
- `Complex`: re-export of `num_complex::Complex`.

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

Usage:

```rust
use physics_in_parallel::math::prelude::*;

let x = 3.0_f64;
let y = x.sqrt();
let z: i64 = x.cast_to();
let c = Complex::new(1.0, 2.0);
let r = c.norm_sqr_real();
```

### Rank-N Tensor

Purpose:

`Tensor<T, Backend>` is the general tensor API. It hides dense/sparse storage details behind a consistent user-facing type while still preserving backend type where operations are type-preserving.

Core structs and traits:

- `Tensor<T, DenseBackend>`: dense rank-N tensor.
- `Tensor<T, SparseBackend>`: sparse rank-N tensor.
- `TensorTrait`: shared tensor behavior.
- `Backend`, `DenseBackend`, `SparseBackend`: backend markers.
- `TensorResult`, `TensorError`: tensor error types.

Core creation API:

```rust
Tensor::<T, DenseBackend>::empty(shape)
Tensor::<T, DenseBackend>::from_vec(shape, data)
Tensor::<T, SparseBackend>::empty(shape)
Tensor::<T, SparseBackend>::from_vec(shape, data)
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

Conversion and explicit-type API:

```rust
tensor.try_cast_to::<U>()
tensor.cast_to::<U>()
tensor.to_dense()
tensor.to_sparse()
```

Linear algebra API:

```rust
tensor.dot(&rhs)
tensor.matmul(&rhs)
tensor.cross(&rhs)
tensor.wedge(&rhs)
tensor.norm_sqr()
```

Usage:

```rust
use physics_in_parallel::math::prelude::*;

let a = Tensor::<f64, DenseBackend>::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
let b = a.scalar_mul(2.0);
let c = a.matmul(&b);
```

### Tensor Random Fillers

Purpose:

`TensorRandFiller` fills dense tensor storage in parallel. It is reused by tensor tests, vector-list random generators, and space coordinate generation.

Core structs and enums:

- `TensorRandFiller`
- `RandType`
- `RngKind`
- `TensorRandError`

Core API:

```rust
TensorRandFiller::new(rand_type, num_rngs)
TensorRandFiller::new_with_rng_kind(rand_type, num_rngs, rng_kind)
filler.refresh(tensor)
filler.try_refresh(tensor)
filler.rng_kind()
```

Random distribution API:

```rust
RandType::Uniform { low, high }
RandType::UniformInt { low, high }
RandType::Normal { mean, std }
RandType::Bernoulli { p }
```

Usage:

```rust
use physics_in_parallel::math::prelude::*;

let mut tensor = dense::Tensor::<f64>::empty(&[1_000, 3]);
let mut filler = TensorRandFiller::new(
    RandType::Normal { mean: 0.0, std: 1.0 },
    Some(4),
);
filler.refresh(&mut tensor);
```

### Matrix

Purpose:

`Matrix<T, Backend>` is a rank-2 wrapper that gives matrix-specific access and operations while reusing rank-N tensor infrastructure internally.

Core structs and traits:

- `Matrix<T, Backend>`
- `MatrixBackend`
- `DenseMatrix<T>`
- `SparseMatrix<T>`
- `DiagonalMatrix<T>`
- `SymmetricMatrix<T>`
- `AntiSymmetricMatrix<T>`
- `UpperTriangularMatrix<T>`
- `LowerTriangularMatrix<T>`
- `StrictUpperTriangularMatrix<T>`
- `StrictLowerTriangularMatrix<T>`

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
```

Core access API:

```rust
matrix.shape()
matrix.rows()
matrix.cols()
matrix.get(i, j)
matrix.set(i, j, value)
matrix.fill(value)
matrix.print()
```

Matrix operations:

```rust
matrix.add(&rhs)
matrix.sub(&rhs)
matrix.elem_mul(&rhs)
matrix.elem_div(&rhs)
&matrix + &rhs
&matrix - &rhs
matrix.transpose()
matrix.hermitian_transpose()
matrix.trace()
matrix.matmul(&rhs)
matrix.scalar_mul(scalar)
matrix.try_cast_to::<U>()
matrix.cast_to::<U>()
matrix.to_dense()
matrix.to_dense_matrix()
```

Usage:

```rust
use physics_in_parallel::math::prelude::*;

let a = DenseMatrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
let b = a.transpose();
let c = a.matmul(&b);
```

### VectorList

Purpose:

`VectorList<T>` stores many fixed-length vectors as a dense rank-N tensor with logical shape `[num_vectors, dim]`. Users work with whole vectors as the natural unit.

Core structs and traits:

- `VectorList<T>`
- `VectorListRand`
- `HaarVectors`
- `NNVectors`

Core creation API:

```rust
VectorList::<T>::empty(dim, num_vectors)
VectorList::<T>::zeros(dim, num_vectors)
VectorList::<T>::from_vec(dim, num_vectors, data)
VectorList::<T>::from_fn(dim, num_vectors, f)
```

Core access API:

```rust
vectors.dim()
vectors.num_vectors()
vectors.shape()
vectors.get(i, axis)
vectors.set(i, axis, value)
vectors.get_vec(i)
vectors.get_vec_mut(i)
vectors.set_vec(i, values)
vectors.axis(axis)
vectors.print()
```

Vector operations:

```rust
vectors.fill(value)
vectors.par_for_each_vec(...)
vectors.par_for_each_vec_mut(...)
vectors.scale_vectors_by_list(scales)
vectors.normalize()
vectors.norms_real()
vectors.add(&rhs)
vectors.sub(&rhs)
vectors.elem_mul(&rhs)
vectors.elem_div(&rhs)
&vectors + &rhs
&vectors - &rhs
vectors.scalar_mul(scalar)
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

Usage:

```rust
use physics_in_parallel::math::prelude::*;

let mut vectors = VectorList::<f64>::empty(3, 2);
vectors.set_vec(0, &[1.0, 0.0, 0.0]);
vectors.set_vec(1, &[0.0, 1.0, 0.0]);

let mut haar = HaarVectors::new(3, 10_000, Some(4));
haar.refresh();
```

### Math IO

Purpose:

`math::io` provides conversion and serialization support for tensors, matrices, and vector lists.

Core APIs:

```rust
serde_json::to_string_pretty(&value)
serde_json::from_str::<T>(json)
value.to_ndarray()
T::from_ndarray(&array)
```

Flat JSON schema:

```json
{
  "kind": "...",
  "shape": [2, 3],
  "data": [1, 2, 3, 4, 5, 6]
}
```

Common math payload kinds:

- `tensor`
- `tensor_sparse`
- `matrix`
- `vector_list`

## Space

Purpose:

`space` adds physical coordinate semantics on top of math data structures. The ready user-facing space implementation is the square lattice.

Common import:

```rust
use physics_in_parallel::space::prelude::*;
```

### Space Trait

Purpose:

`Space<T>` gives simulation code one common interface for reading, mutating, filling, and saving spatial containers.

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

For `SquareLattice`, coordinates are signed. Raw coordinates outside the shape are interpreted by the lattice boundary condition.

### SquareLattice

Purpose:

`SquareLattice<T>` represents a square, cubic, or hypercubic lattice over a tensor-style shape such as `[128]`, `[64, 64]`, or `[32, 64, 16]`.

Core structs and enums:

- `SquareLattice<T>`
- `SquareLatticeConfig`
- `SquareLatticeInitMethod<T>`
- `BoundaryCondition`
- `VacancyValue`

Boundary conditions:

```rust
BoundaryCondition::Periodic
BoundaryCondition::Reflective
```

Config API:

```rust
SquareLatticeConfig::new(shape, boundary)
SquareLatticeConfig::periodic(shape)
SquareLatticeConfig::reflective(shape)
cfg.shape()
cfg.rank()
cfg.num_sites()
cfg.tensor_shape()
```

Initialization API:

```rust
SquareLatticeInitMethod::Empty
SquareLatticeInitMethod::Uniform { val }
SquareLatticeInitMethod::RandomUniformChoices { choices }
SquareLatticeInitMethod::SeededCenter { val }
```

Lattice API:

```rust
SquareLattice::<T>::new(cfg, init_method)
SquareLattice::<T>::vacancy()
lattice.data()
lattice.set_vacant(coord)
lattice.is_vacant(coord)
lattice.fill_vacancy()
lattice.downsample(target_shape)
lattice.rescale(target_shape)
```

Space API on square lattices:

```rust
lattice.get(coord)
lattice.get_mut(coord)
lattice.set(coord, value)
lattice.set_all(value)
lattice.dims()
lattice.linear_size()
```

Usage:

```rust
use physics_in_parallel::space::prelude::*;

let cfg = SquareLatticeConfig::new(&[64, 64], BoundaryCondition::Periodic);
let mut lattice = SquareLattice::<usize>::new(
    cfg,
    SquareLatticeInitMethod::Uniform { val: 1 },
);

lattice.set(&[-1, 0], 7);
assert_eq!(*lattice.get(&[63, 0]), 7);
```

### Square-Lattice Kernels

Purpose:

Square-lattice kernels describe how random displacements are sampled for lattice pair generation.

Core structs and enums:

- `KernelType`
- `Kernel`
- `PowerLawKernel`
- `UniformKernel`
- `NearestNeighborKernel`

Kernel types:

```rust
KernelType::NearestNeighbor { d }
KernelType::Uniform { c, l }
KernelType::PowerLaw { c, l, mu }
```

Core API:

```rust
create_kernel(kernel_type)
kernel.sample(n)
kernel.kind()
```

Usage:

```rust
use physics_in_parallel::space::prelude::*;

let kernel = create_kernel(KernelType::PowerLaw {
    c: 1.0,
    l: 20.0,
    mu: 2.0,
});
let lengths = kernel.sample(1_000);
```

### RandPairGenerator

Purpose:

`RandPairGenerator` creates batches of source coordinates, displacement vectors, and raw target coordinates for square-lattice workflows.

Important responsibility split:

- Sources are generated inside the provided shape.
- Displacements are raw move vectors.
- Targets are raw `source + displacement`.
- Boundary interpretation is done later by `SquareLattice::get`, `set`, or `get_mut`.

Core structs and enums:

- `RandPairGenerator`
- `SourceMode`

Source modes:

```rust
SourceMode::Origin
SourceMode::RandomUniform
SourceMode::CustomFiller(filler)
```

Core API:

```rust
RandPairGenerator::new(shape, kernel_type, num_pairs, source_mode, num_rngs)
gen.refresh()
gen.refresh_sources()
gen.refresh_displacements()
gen.refresh_targets()
gen.shape()
gen.rank()
gen.num_pairs()
gen.kernel_type()
gen.sources()
gen.displacements()
gen.targets()
gen.source(i)
gen.displacement(i)
gen.target(i)
```

Usage:

```rust
use physics_in_parallel::space::prelude::*;

let mut generator = RandPairGenerator::new(
    &[64, 64],
    KernelType::NearestNeighbor { d: 2 },
    10_000,
    SourceMode::RandomUniform,
    Some(4),
);

generator.refresh();

let source = generator.source(0);
let target = generator.target(0);
```

### Space IO

Purpose:

`space::io` contains IO behavior for space types. Current ready IO support is for square lattices.

Core API:

```rust
save_square_lattice(&lattice, target_shape, path)
serde_json::to_string_pretty(&lattice)
serde_json::from_str::<SquareLattice<T>>(json)
SquareLattice::from_ndarray(&array, boundary)
lattice.to_ndarray()
lattice.serialize()
```

Square-lattice JSON kinds:

- `square_lattice_periodic`
- `square_lattice_reflective`

Usage:

```rust
use physics_in_parallel::space::prelude::*;

let cfg = SquareLatticeConfig::periodic(&[4, 4]);
let lattice = SquareLattice::<usize>::new(
    cfg,
    SquareLatticeInitMethod::Uniform { val: 2 },
);

let json = serde_json::to_string_pretty(&lattice).unwrap();
```

## Engines

Purpose:

`engines` provides model-agnostic runtime infrastructure. The ready backend is `soa`, a structure-of-arrays layout for object attributes and interaction payloads.

Common import:

```rust
use physics_in_parallel::engines::prelude::*;
```

### AttrsMeta

Purpose:

`AttrsMeta` stores human-facing metadata for a `PhysObj`.

Core API:

```rust
AttrsMeta::empty()
AttrsMeta::new(id, label, comment)
meta.serialize()
```

### AttrsCore

Purpose:

`AttrsCore` stores named typed attribute columns. Each column is a `VectorList<T>` with shape `[n_objects, dim]`. Different labels may store different scalar types, but all labels in one core must have the same `n_objects`.

Core structs and enums:

- `AttrsCore`
- `AttrsError`
- `AttrId`

Core API:

```rust
AttrsCore::empty()
core.len()
core.is_empty()
core.contains(label)
core.n_objects()
core.labels()
core.id_of(label)
core.label_of(id)
core.allocate::<T>(label, dim, n_objects)
core.insert(label, vector_list)
core.remove(label)
core.rename(from, to)
core.get::<T>(label)
core.get_mut::<T>(label)
core.get_by_id::<T>(id)
core.get_by_id_mut::<T>(id)
core.vector_of::<T>(label, obj)
core.vector_of_mut::<T>(label, obj)
core.set_vector_of::<T>(label, obj, value)
core.dim_of(label)
core.dim_of_id(id)
core.type_name_of(label)
core.type_name_of_id(id)
core.serialize()
```

The label-based API is the normal user path. Attribute IDs are generated
automatically and are available as an optional expert path for repeated lookup.

Usage:

```rust
use physics_in_parallel::engines::prelude::*;

let mut core = AttrsCore::empty();
core.allocate::<f64>("r", 3, 1_000).unwrap();
core.set_vector_of::<f64>("r", 0, &[1.0, 2.0, 3.0]).unwrap();
let r0 = core.vector_of::<f64>("r", 0).unwrap();
```

### PhysObj

Purpose:

`PhysObj` bundles `AttrsMeta` and `AttrsCore` into one serializable object collection.

Core API:

```rust
PhysObj::empty()
PhysObj::new(meta, core)
obj.serialize()
obj.save_to_json(output_dir, filename)
```

Usage:

```rust
use physics_in_parallel::engines::prelude::*;

let meta = AttrsMeta::new(0, "particles", "particle state");
let mut core = AttrsCore::empty();
core.allocate::<f64>("v", 3, 100).unwrap();

let obj = PhysObj::new(meta, core);
let json = obj.serialize().unwrap();
```

### InteractionTopology

Purpose:

`InteractionTopology` maps object-node lists to stable interaction IDs. It supports mixed arity, so an interaction can be a pair, a triple, or a longer n-body term.

`InteractionOrder::Ordered` preserves caller order, so `[i, j]` and `[j, i]` are different interactions.

`InteractionOrder::Unordered` sorts the node list before lookup, so `[i, j]` and `[j, i]` identify the same symmetric interaction.

Core structs and enums:

- `InteractionTopology`
- `InteractionNodes`
- `InteractionOrder`
- `ObjId`
- `InteractionId`

Core API:

```rust
InteractionTopology::new(n_objects)
InteractionTopology::with_order(n_objects, order)
topology.n_objects()
topology.order()
topology.set_order(order)
topology.set_n_objects(n_objects)
topology.prune_n_objects(n_objects)
topology.len()
topology.is_empty()
topology.capacity_slots()
topology.free_slot_count()
topology.nodes_of(id)
topology.id_of(nodes)
topology.contains(nodes)
topology.add(nodes)
topology.remove(nodes)
topology.clear()
topology.iter()
topology.id_of_pair(i, j)
topology.add_pair(i, j)
topology.remove_pair(i, j)
```

Interaction order:

```rust
InteractionOrder::Ordered
InteractionOrder::Unordered
```

### Interaction

Purpose:

`Interaction<T>` combines an `InteractionTopology` with payload storage for one uniform payload type. This is the normal user-facing container for model interactions because topology edits made through `Interaction<T>` keep payload storage synchronized.

Core API:

```rust
Interaction::<T>::new(n_objects, order)
Interaction::<T>::with_topology(topology)
interaction.topology()
interaction.topology_mut()
interaction.len()
interaction.is_empty()
interaction.n_objects()
interaction.order()
interaction.set_order(order)
interaction.set_n_objects(n_objects)
interaction.prune_n_objects(n_objects)
interaction.contains(nodes)
interaction.set(nodes, payload)
interaction.remove(nodes)
interaction.get(nodes)
interaction.get_mut(nodes)
interaction.set_pair(i, j, payload)
interaction.get_pair(i, j)
interaction.get_pair_mut(i, j)
interaction.remove_pair(i, j)
interaction.clear()
interaction.iter()
interaction.par_for_each_payload(...)
interaction.par_for_each_payload_mut(...)
interaction.par_for_each(...)
```

Usage:

```rust
use physics_in_parallel::engines::prelude::*;

let mut springs = Interaction::<f64>::new(10, InteractionOrder::Unordered);
let id = springs.set_pair(0, 1, 1.5).unwrap();
assert_eq!(*springs.get(&[1, 0]).unwrap().unwrap(), 1.5);
```

### NeighborList

Purpose:

`NeighborList` is a cell-linked candidate-pair generator for bounded particle systems. It partitions a continuous box into cells and emits unique unordered nearby-cell candidate pairs. It does not apply the final physical distance cutoff; model wrappers do that filtering.

Core structs and enums:

- `NeighborList`
- `NeighborListError`

Core API:

```rust
NeighborList::new(min, max, cell_width)
neighbor_list.dim()
neighbor_list.min()
neighbor_list.max()
neighbor_list.cell_width()
neighbor_list.cells_per_axis()
neighbor_list.num_cells()
neighbor_list.num_objects()
neighbor_list.clear()
neighbor_list.rebuild(positions, n_objects)
neighbor_list.for_each_pair_candidate(|i, j| { ... })
neighbor_list.collect_pair_candidates()
```

Usage:

```rust
use physics_in_parallel::engines::prelude::*;

let mut list = NeighborList::new(&[0.0, 0.0], &[10.0, 10.0], 1.0).unwrap();
let positions = vec![0.1, 0.2, 0.4, 0.5];
list.rebuild(&positions, 2).unwrap();
list.for_each_pair_candidate(|i, j| {
    println!("{i} {j}");
});
```

## Higher-Level Modules

The crate also contains:

- `models`: concrete model packages.

`models` is outside the current README focus and should be documented after its next correctness and API review.

## Examples

Runnable examples:

```bash
cargo run --example serde_flat_json
cargo run --example vector_list_ndarray
cargo run --example tensor_rand_large_benchmark --release
cargo run --example vector_list_haar_benchmark --release
```

## Verification

The current reviewed math, space, and engines APIs are covered by:

```bash
cargo test --test math --test space --test engines
cargo test
cargo doc --no-deps
```

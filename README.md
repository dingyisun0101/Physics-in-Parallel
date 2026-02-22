# Physics-in-Parallel

Physics-in-Parallel is a Rust package for high-performance numerical simulation with an explicit layered architecture:

`math -> space -> engines -> models`

The main design goal is to keep low-level numeric infrastructure generic and reusable, then build domain-specific simulation tools on top of it.

---

## Architecture at a glance

### Layer order
1. `math`: scalar + tensor foundations.
2. `space`: domain and kernel abstractions built on `math`.
3. `engines`: generic simulation runtime primitives (currently SoA-focused) built on `math` and usable with `space`.
4. `models`: ready-to-use physical model packages built on `engines`.

### Top-level user-facing modules

These are exposed from `src/lib.rs`:

- `physics_in_parallel::math`
- `physics_in_parallel::space`
- `physics_in_parallel::engines`
- `physics_in_parallel::models`
- `physics_in_parallel::prelude`

`prelude` currently re-exports the crate-level math/space/engines preludes.

---

## Module details

## 1) `math`

### Purpose
`math` is the numeric foundation layer. It provides the scalar and tensor machinery used by all higher layers.

### How it is implemented

- Scalar abstraction:
  - A unified `Scalar` trait supports integers, real floats, and complex numbers behind one API.
  - Common scalar operations (conversion, norms, conjugation-style behavior, finiteness checks) are centralized here.

- Tensor core:
  - Dense backend: contiguous flat storage for cache-friendly bulk operations.
  - Sparse backend: sparse representation optimized for low occupancy patterns.
  - Unified front type: `Tensor<T, Backend>` with backend-specialized behavior under one interface.
  - Core operations are parallelized where appropriate using Rayon.

- Rank-2 math:
  - `Tensor2D` for 2D tensor views/operations.
  - `Matrix` for matrix-centric workflows and matrix traits.
  - `VectorList` for "many vectors with fixed dimension" storage and operations.

- Random fillers:
  - `TensorRandFiller` and `RandType` support random initialization patterns reused by model code.
  - Vector-list random generators (`HaarVectors`, `NNVectors`) build on the same tensor infrastructure.

### User entry points

- `physics_in_parallel::math::*` for module-level APIs.
- `physics_in_parallel::math::prelude::*` for common math imports.

---

## 2) `space`

### Purpose
`space` adds geometric and domain semantics on top of `math` tensors. It represents simulation domains and spatial kernels without tying to a particular physical model.

### How it is implemented

- `Space` trait:
  - Shared abstraction for space/domain backends.

- Kernel module:
  - Common kernel types (`NearestNeighbor`, `PowerLaw`, `Uniform`) via `KernelType`.
  - Concrete kernel implementations and kernel factory function (`create_kernel`).

- Discrete representations:
  - Grid configuration (`GridConfig`) and initialization (`GridInitMethod`).
  - `Grid<T>` storage representation for lattice-like spaces.
  - Serialization helpers (`save_grid`) and vacancy conventions.
  - Random pair generation utilities (`RandPairGenerator`) for stochastic displacement/workflows.

### How it builds on `math`

All concrete space data structures are implemented using math-layer tensor/scalar facilities. `space` does not duplicate numeric kernels; it composes them.

### User entry points

- `physics_in_parallel::space::*` for module-level APIs.
- `physics_in_parallel::space::prelude::*` for common space imports.

---

## 3) `engines`

### Purpose
`engines` provides model-agnostic runtime infrastructure for simulation state and interaction management.

Current primary implementation is `engines::soa`.

### How it is implemented

- `PhysObj` data container:
  - Built from `AttrsMeta` + `AttrsCore`.
  - `AttrsCore` stores heterogeneous typed columns keyed by attribute label.
  - Each attribute column is a typed `VectorList<T>` stored behind runtime-erased trait objects (`DynVectorList`), allowing mixed scalar types.
  - Enforces shape consistency (`n_objects`, per-attribute dimension checks) through `AttrsError`.

- Interaction backend:
  - `Topology`: maps mixed-arity interaction keys (e.g., `(i,j)` or `(i,j,k,...)`) to stable slot IDs.
  - Supports directed or undirected validation policy.
  - Uses hole reuse for cheap insert/delete churn.
  - `Interaction<T>` combines topology with hidden payload storage for uniform payload type `T`.
  - Exposes key-based insert/get/remove APIs and parallel payload iteration.

### How it builds on previous layers

- Builds directly on `math` data structures (`VectorList`) for SoA storage.
- Designed to work with `space` outputs (neighbor structures, kernels) but remains generic and model-independent.

### User entry points

- `physics_in_parallel::engines::soa::*`
- `physics_in_parallel::engines::prelude::*`

---

## 4) `models`

### Purpose
`models` is the domain package layer: concrete simulation model modules built from the reusable engine + math stack.

Current package: `models::particles`.

### How `models::particles` is implemented

- `attrs`:
  - Canonical attribute labels (`r`, `v`, `a`, `m`, `m_inv`, `alive`) for consistent model code.

- `create_state`:
  - `create_template(dim, num_particles)` builds a particle `PhysObj` with canonical fields.
  - `randomize_r(...)` and `randomize_v(...)` provide state initialization methods.
  - Uses math random fillers + Rayon parallel transforms for bulk initialization.

- `integrator`:
  - Integrator trait and Euler-family implementations for time stepping.
  - Operates directly on `PhysObj` attributes.
  - Uses parallel chunk-wise updates over vector-list storage.

- `boundary`:
  - Boundary trait and concrete boundary conditions (periodic, clamp, reflect).
  - Works directly on `PhysObj` canonical fields (`r`, `v`, optional `alive`).
  - Reflect boundary uses a dual-pass strategy to preserve mutable alias safety while remaining parallel.

- `thermostat`:
  - Thermostat trait and Langevin implementation.
  - Updates velocity fields from target temperature/friction parameters.
  - Validates state shape/physics constraints and applies stochastic updates.

### How it builds on previous layers

- `models` is where engine-generic containers (`PhysObj`, `Interaction`) become concrete physics workflows.
- Random initialization, vector math, and parallelism are inherited from `math`.
- State and interaction organization is inherited from `engines`.
- Optional domain logic can be composed with `space`.

### User entry points

- `physics_in_parallel::models::particles::...`
  - `attrs`
  - `create_state`
  - `integrator`
  - `boundary`
  - `thermostat`

---

## Typical usage flow

1. Use `math` to define numeric types, tensor operations, and random generators.
2. Use `space` to describe domain/kernels when your model needs geometric structure.
3. Use `engines` to hold simulation state (`PhysObj`) and interaction topology/payloads (`Interaction<T>`).
4. Use `models` packages to run concrete workflows (create state, integrate, apply boundaries/thermostats, observe).

This layering keeps the low-level infrastructure reusable while letting models stay concise and domain-focused.

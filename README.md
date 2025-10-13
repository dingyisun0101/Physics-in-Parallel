# Physics-in-Parallel

**Physics in Parallel** is a scientific computing package in Rust for **massive, memory-bound simulations** where *full vectorization isn’t feasible*. It targets workloads that exceed cache and RAM budgets, need irregular access patterns, or are best served by a **hybrid SoA/AoS** approach with **block-wise/streaming** execution.

---

## Core Goals
1. **Unified Scalar Abstraction**  
   A single `Scalar` trait unifies integers, floats, and complex types with a consistent API (conjugation, norms, sqrt, finiteness).

2. **Tensor Infrastructure (Hybrid-layout Friendly)**  
   - Parallel elementwise ops (`+`, `-`, `*`, `/`) via Rayon.
   - **Tile/block iterators** and **zero-copy views** designed to cooperate with SoA and AoS neighbors.
   - Human-readable serialization (JSON, DSL strings).
   - Multiple backends availible:
      - Dense:
         - flat-memory tensors for cache efficiency.  
      - Sparse: 
         - Sparse and symmetry-aware storage backed by hashset.

2. **2D Tensor Infrastructure (Matrix + VectorList)**
   - Matrix: 
      - A wrapper for tensor that adds linalg methods support.
      - Allow slicing and viewing by row or col.
   - VectorList:
      - A wrapper for Matrix.
      - Each col represents a vector.
      - Allow bulk operations on vectors (normalization, ...)
      

3. **Random Tensor Generation**  
   Efficient, **chunked** generators for random vectors (uniform, Gaussian, etc.), supporting **streamed/partial fills** for Monte Carlo and stochastic dynamics on large states.

4. **Space**
   Provide abstraction for space representations.
   - Dense:
      - A wrapper for dense tensor. The tensor represents a square lattice with periodic boundary.
      - Each site is accessible through a cartesion coordinate.
   - Sparse:
      - A wrapper for sparse tensor. It is best for describing continuous space containing finite elements.

5. **Parallelism by Default (Under Memory Constraints)**  
   - Rayon-powered **tile-level** parallelism to respect cache lines and NUMA.  
   - **Block-wise** maps/zips, fused passes, and streaming pipelines to reduce materialization.  
   - Scales from laptops to HPC nodes **without requiring full in-RAM tensors**.

---

## Why Hybrid SoA/AoS?
Some physics tasks (e.g., lattice kernels with long tails, neighbor lists, event-driven updates, or out-of-core grids) don’t map cleanly to a single layout or to global vectorization. **Physics in Parallel** embraces:
- **SoA** for bandwidth-bound sweeps (norms, reductions, scalar transforms).  
- **AoS** (or AoS-like views) where locality across coordinates simplifies kernels or boundary logic.  
- **Block/tiling** to operate on windows that fit in cache, enabling halos/ghost cells and sliding updates.  
- **Streaming/randomized passes** to amortize RNG and avoid full materialization.

---

## Use Cases
- **Stochastic field dynamics:** diffusion + Langevin noise on very large grids using streaming tiles.  
- **Ecology & population dynamics:** lattice models with structured, possibly long-range dispersal kernels on domains too large for global vectorization.  
- **General numerical physics:** PDE/ODE solvers, interacting-particle systems, and kernels mixing SoA scans with AoS-style neighbor ops.  

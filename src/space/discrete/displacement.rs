// src/sim/sampling/rand_pairs.rs
/*!
Random **source–target** pair generation for lattice-based simulations.

This module wires together:
- a **direction sampler** (Haar or neighborhood) from `vector_list_rand`,
- a **step-length sampler** from a pluggable `Kernel`,
- and a simple **composition rule**: `target = source + displacement`.

It follows the annotated, invariants-first style used across the codebase.

# High-level overview

1. (Optional) **Randomize sources**: overwrite `source_coords` with uniform integers.
2. **Sample displacements** with the provided `kernel`:
   - `PowerLaw` / `Uniform`: sample **unit directions** (Haar) and **norms**, then scale.
   - `NearestNeighbor`: sample **encoded neighbor indices** and decode to vectors.
3. **Compose targets**: `target_coords = source_coords + displacements`.

> **Note**
> - The **units** of `source_coords` and the **interpretation** of `norms` / displacements
>   are your model contract. For discrete lattices you typically cast/round displacement
>   components to integer steps (as we do below via `cast_to::<isize>()`).
> - The “randomize sources” branch currently samples from `[0, num_vectors)` for **each
>   coordinate component**. If your lattice side length is `l`, you likely want `[0, l)`
>   instead; swap the upper bound accordingly.

*/

use super::super::super::math::tensor::{
    tensor_rand::{RandType, fill_random},
};
use super::super::super::math::vector::vector_list::VectorList;
use super::super::super::math::vector::vector_list_rand::{
    gen_haar_vectors,        // random unit directions in R^dim (SoA VectorList)
    gen_neighborhood_vectors // map encoded neighbor IDs → displacement basis vectors
};

use super::super::kernel::*; // Kernel, KernelType, create_kernel


// ================================================================================================
// -------------------- Random Displacement Source - Target Pair Generator ------------------------
// ================================================================================================

/**
Stateful generator that owns a **sampling kernel** and can populate
`(source, target)` index/coordinate pairs in bulk.

- `kernel` is a trait object so you can switch implementations at runtime.
- The generator itself is **thin**; heavy lifting is delegated to helpers.

# Invariants
- The `kernel` must outlive this struct (owned `Box<dyn Kernel>` satisfies this).
- The caller ensures that the **shape** of `source_coords` and `target_coords` is consistent
  (same `dim` and `num_vectors`), though we also check at runtime in the helper.

# Typical usage
- Construct via [`RandPairGenerator::new`].
- Call [`RandPairGenerator::random`] repeatedly to refresh `(source, target)` buffers.
*/

#[derive(Clone)]
pub struct RandPairGenerator {
    kernel: Box<dyn Kernel>,
    pub source_coords_cache: VectorList<isize>,
    pub target_coords_cache: VectorList<isize>,
}

impl RandPairGenerator {
    /// Build a generator by **constructing the kernel** from its [`KernelType`].
    ///
    /// This is a convenience wrapper around `create_kernel(kernel_type)`.
    #[inline]
    pub fn new(kernel_type: KernelType, dim: usize, num_pairs: usize) -> Self {
        let kernel = create_kernel(kernel_type);
        let source_coords_cache: VectorList<isize> = VectorList::new(dim, num_pairs);
        let target_coords_cache = source_coords_cache.clone();
        Self { kernel, source_coords_cache, target_coords_cache }
    }

    /**
    Populate `(source_coords, target_coords)` with a batch of random pairs.

    - If `randomize_source == true`, `source_coords` are overwritten uniformly.
    - Displacements are then drawn from `self.kernel` and **added** to sources.
    - `target_coords` are **replaced** with the resulting sum.

    ## Parameters
    - `source_coords`: SoA list of `num_vectors` source coordinates (each `dim`-D).
    - `target_coords`: SoA output buffer receiving targets (must be same shape).
    - `randomize_source`: if `true`, overwrite `source_coords` with random integers.

    ## Panics
    - If `source_coords` and `target_coords` have mismatched `dim` or `num_vectors`.

    ## Complexity
    - O(`num_vectors * dim`) with vectorized/SoA arithmetic under the hood.

    ## Notes
    - The source randomization range is `[0, num_vectors)`. If your lattice side-length is `l`,
      use `[0, l)` instead by changing the `RandType::UniformInt` upper bound.
    */
    #[inline]
    pub fn random(
        &mut self,
        randomize_source: bool,
    ) {
        gen_random_idx_pairs_by_kernel(
            self.kernel.as_ref(),
            &mut self.source_coords_cache,
            &mut self.target_coords_cache,
            randomize_source,
        );
    }
}


// ================================================================================================
// ----------------------------------- Generator Helpers ------------------------------------------
// ================================================================================================

/**
Core routine implementing the **three-step** sampling pipeline:

1. **(Optional) Source randomization**  
   If `randomize_source`, fill `source_coords` with integers in `[0, num_vectors)`.

2. **Displacement sampling (depends on kernel)**  
   - `PowerLaw` / `Uniform`:
     - Draw `num_vectors` **unit direction vectors** via `gen_haar_vectors(dim, num_vectors)`.
     - Draw `num_vectors` **norms** via `kernel.sample(num_vectors)`.
     - Scale each unit vector by its corresponding norm (broadcasted as per-vector scalar).
     - Cast the result to `isize` coordinates for discrete displacements.
   - `NearestNeighbor`:
     - Draw `num_vectors` **encoded neighbor IDs** via `kernel.sample(num_vectors)`.
     - Decode to one-step displacement vectors via `gen_neighborhood_vectors`.

3. **Compose targets**  
   `target_coords = source_coords + displacements` (SoA elementwise vector add).

## Parameters
- `kernel`: sampling strategy (power-law tails, uniform, or nearest-neighbor).
- `source_coords`: mutable list of sources (possibly randomized in-place).
- `target_coords`: output list of targets (overwritten).
- `randomize_source`: toggle for step (1).

## Contract & Panics
- Assumes `source_coords.dim() == target_coords.dim()` and
  `source_coords.num_vectors() == target_coords.num_vectors()`.
  This is **checked** and will panic on mismatch.

## Numerical notes
- Casting floating displacements to `isize` relies on your `VectorList::cast_to::<isize>()`
  semantics (commonly truncation toward zero). Consider explicit rounding if needed.
*/
#[inline]
pub fn gen_random_idx_pairs_by_kernel(
    kernel: &dyn Kernel,
    source_coords: &mut VectorList<isize>,
    target_coords: &mut VectorList<isize>,
    randomize_source: bool,
) {
    // ----------------------- Shape checks -----------------------
    let dim = source_coords.dim();
    let num_vectors = source_coords.num_vectors();
    assert_eq!(
        target_coords.dim(), dim,
        "dim mismatch: sources are {}-D, targets are {}-D", dim, target_coords.dim()
    );
    assert_eq!(
        target_coords.num_vectors(), num_vectors,
        "batch mismatch: sources have {}, targets have {}",
        num_vectors, target_coords.num_vectors()
    );

    // ----------------------- (1) Sources ------------------------
    // Optionally randomize sources IN-PLACE.
    //
    // NOTE: This uses [0, num_vectors) as the inclusive range for **each coordinate
    // component**. If your lattice side length is `l`, you likely want [0, l) instead:
    //   RandType::UniformInt { low: 0, high: (l as i64) }
    if randomize_source {
        fill_random(
            &mut source_coords.data,
            RandType::UniformInt { low: 0, high: num_vectors as i64 }
        );
    }

    // -------------------- (2) Displacements ---------------------
    let displacements: VectorList<isize> = match kernel.kind() {
        // For continuous kernels, directions × norms:
        KernelType::PowerLaw { .. } | KernelType::Uniform { .. } => {
            // A) sample unit directions (Haar on S^{dim-1})
            let unit_vectors = gen_haar_vectors(dim, num_vectors);
            // B) sample step lengths (scalar per vector)
            let norms = kernel.sample(num_vectors);
            // C) scale each unit vector by its corresponding norm
            //    and cast to discrete steps (isize)
            unit_vectors.scale_by_list(&norms).cast_to::<isize>()
        }
        // For discrete nearest-neighbor kernels, decode IDs → basis vectors.
        KernelType::NearestNeighbor { .. } => {
            let encoded = kernel.sample(num_vectors);
            gen_neighborhood_vectors(dim, num_vectors, encoded)
        }
    };

    // ---------------------- (3) Compose -------------------------
    // targets = sources + displacements
    let sum = source_coords.clone() + displacements;
    *target_coords = sum;
}
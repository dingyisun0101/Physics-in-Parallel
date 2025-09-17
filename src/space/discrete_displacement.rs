use super::super::math::tensor::{ dense::Tensor, tensor_rand:: { fill_random_f64, RandType } };
use super::super::math::vector::{vector_list::VectorList};
use super::super::math::vector::vector_list_rand::{ fill_haar_vecs };

use super::discrete::GridConfig;
use super::kernel::*;


// ================================================================================================
// -------------------------- Random Source & Target Pair Generator Module ------------------------
// ================================================================================================

pub struct RandPairGenerator {
    grid_cfg: GridConfig,
    kernel: Box<dyn Kernel>,
}

impl <const D: usize> RandPairGenerator {
    pub fn new(grid_cfg: GridConfig, kernel_type: KernelType) -> Self {
        let kernel = create_kernel(kernel_type);
        Self { grid_cfg, kernel }
    }

    pub fn random(&self, num_pairs: usize) -> (VectorList<usize, D>, VectorList<usize, D>) {
`           gen_random_idx_pairs_by_kernel(
            num_pairs,
            self.kernel.as_ref(),
            self.grid_cfg.d,
            self.grid_cfg.l,
        )
    }
}

// ----------------------------- Generator Helpers ----------------------------
pub fn <const D: usize> gen_random_idx_pairs_by_kernel(
    num_pairs: usize,
    kernel: &dyn Kernel,
    d: usize,
    l: usize,
) -> (VectorList<usize, D>, VectorList<usize, D>) {
    assert!(d > 0 && num_pairs > 0, "Invalid dimensions");

    // 1) Sources in [0, l) (float for now; we'll floor->usize at the end)
    let sources = RandVectors::<f64>::new(
        d,
        num_pairs,
        RandType::ElementWiseUniform(0.0, l as f64),
    );

    // 2) Random directions (Gaussian), then normalize to unit vectors
    let mut directions = RandVectors::<f64>::new(d, num_pairs, RandType::ElementWiseNormal);
    directions.normalize();

    // 3) Sample radii with the kernel into a temporary Tensor<f64>, in-place
    //    Shape can be [1, num_pairs] or a flat 1D buffer depending on your Tensor API.
    //    We'll assume a 1D tensor constructor from a flat vec for clarity.
    let mut radii = Tensor::<f64>::from_vec(vec![0.0; num_pairs]); // placeholder zeros
    kernel.sample(&mut radii); // fills radii.data with kernel-distributed distances

    // 4) Rescale directions by sampled radii (expects a slice of length = num_pairs)
    directions.rescale(&radii.data);

    // 5) targets = sources + directions
    let targets = sources.clone() + directions;

    // 6) Convert to integer coordinates via floor
    let sources_usize = sources.to_usize();
    let targets_usize = targets.to_usize();

    (sources_usize, targets_usize)
}

/// Generate random index pairs where each target is a nearest neighbor of its source,
/// with periodic boundary conditions in an l^d lattice.
pub fn gen_random_idx_pairs_by_nn(
    num_pairs: usize,
    d: usize,
    l: usize,
) -> (RandVectors<usize>, RandVectors<usize>) {
    assert!(d > 0 && num_pairs > 0, "Invalid dimensions");

    let num_neighbors = 2 * d;

    // Step 1: Sources in [0, l)^d, shape = [num_pairs, d]
    let sources = RandVectors::<usize>::new(
        d,
        num_pairs,
        RandType::ElementWiseUniform(0, l),
    );
    let mut targets = sources.clone();

    // Step 2: One random direction per pair in [0, 2d)
    let direction_indices = RandVectors::<usize>::new(
        1,
        num_pairs,
        RandType::ElementWiseUniform(0, num_neighbors),
    );

    // Step 3: Displace each target along ±axis (periodic BCs)
    for i in 0..num_pairs {
        let dir = direction_indices.get(i, 0); // 0..2d-1
        let axis = dir / 2;
        let delta = if dir % 2 == 0 { 1 } else { l - 1 }; // +1 or -1 modulo l

        let val = targets.get(i, axis);
        targets.update(i, axis, (val + delta) % l);
    }

    (sources, targets)
}

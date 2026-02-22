/*!
    Random **source–target** pair generation for lattice-based simulations.
    Note that a DENSE backend is used.

    Pipeline:
        1) (optional) randomize sources
        2) sample displacements from `kernel`
        3) compose: targets = sources + displacements
*/

use crate::math::prelude::{
    HaarVectors,
    NNVectors,
    TensorRandFiller,
    VectorList,
    VectorListRand,
};
use crate::space::kernel::*;



// ================================================================================================
// ---------------------------- Displacement cache (enum) -----------------------------------------
// ================================================================================================

#[derive(Clone)]
enum DispCache {
    Haar(HaarVectors), // f64 unit dirs (scaled later, then cast)
    NN(NNVectors),     // isize one-hot ±1
}

// ================================================================================================
// -------------------- Random Displacement Source - Target Pair Generator ------------------------
// ================================================================================================

#[derive(Clone)]
pub struct RandPairGenerator {
    kernel: Box<dyn Kernel>,
    source_coords_filler: Option<TensorRandFiller>,
    pub source_coords_cache: VectorList<isize>,
    pub target_coords_cache: VectorList<isize>,
    displacement_cache: DispCache,
}

impl RandPairGenerator {
    /// Build a generator by constructing the kernel and allocating caches.
    #[inline]
    pub fn new(
        kernel_type: KernelType,
        dim: usize,
        num_pairs: usize,
        source_coords_filler: Option<TensorRandFiller>,
        num_rngs: Option<usize>,
    ) -> Self {
        let kernel = create_kernel(kernel_type);
        let source_coords_cache: VectorList<isize> = VectorList::empty(dim, num_pairs);
        let target_coords_cache = source_coords_cache.clone();

        // Choose displacement cache based on kernel type
        let displacement_cache = match kernel_type {
            KernelType::NearestNeighbor { .. } => DispCache::NN(NNVectors::new(dim, num_pairs, num_rngs)),
            _ => DispCache::Haar(HaarVectors::new(dim, num_pairs, num_rngs)),
        };

        Self {
            kernel,
            source_coords_cache,
            source_coords_filler,
            target_coords_cache,
            displacement_cache,
        }
    }

    /// Populate the internal caches with a new random batch.
    #[inline]
    pub fn refresh(&mut self) {
        gen_random_idx_pairs_by_kernel(
            self.kernel.as_ref(),
            &mut self.source_coords_cache,
            &mut self.source_coords_filler,
            &mut self.target_coords_cache,
            &mut self.displacement_cache,
        );
    }

    /// Accessors if needed downstream
    #[inline] pub fn sources(&self) -> &VectorList<isize> { &self.source_coords_cache }
    #[inline] pub fn targets(&self) -> &VectorList<isize> { &self.target_coords_cache }
}



// ================================================================================================
// ----------------------------------- Generator Helpers ------------------------------------------
// ================================================================================================

/// Core routine: optional source randomization → displacement sampling → compose.
#[inline]
fn gen_random_idx_pairs_by_kernel(
    kernel: &dyn Kernel,
    source_coords: &mut VectorList<isize>,
    source_coords_filler: &mut Option<TensorRandFiller>,
    target_coords: &mut VectorList<isize>,
    displacement_cache: &mut DispCache,
) {
    // ----------------------- Shape checks -----------------------
    let dim = source_coords.dim();
    let num_vectors = source_coords.num_vectors();
    assert_eq!(
        target_coords.dim(),
        dim,
        "dim mismatch: sources are {}-D, targets are {}-D",
        dim,
        target_coords.dim()
    );
    assert_eq!(
        target_coords.num_vectors(),
        num_vectors,
        "batch mismatch: sources have {}, targets have {}",
        num_vectors,
        target_coords.num_vectors()
    );

    // ----------------------- (1) Sources ------------------------
    if source_coords_filler.is_some() {
        if let Some(filler) = source_coords_filler.as_mut() {
            // Fills the **flat** SoA buffer.
            filler.refresh(&mut source_coords.as_tensor_mut());
        }
    }

    // -------------------- (2) Displacements ---------------------
    let displacements: &VectorList<isize> = match (kernel.kind(), displacement_cache) {
        // Continuous kernels: Haar directions × scalar norms → cast to isize steps.
        (KernelType::PowerLaw { .. } | KernelType::Uniform { .. }, DispCache::Haar(hc)) => {
            // a) sample unit directions (Normal + normalize)
            hc.refresh();
            // b) sample step lengths (scalar per vector)
            let norms: Vec<f64> = kernel.sample(num_vectors);
            // c) scale columns by per-vector norms and cast to integer steps
            hc.vl.scale_vectors_by_list(&norms);
            &hc.vl.cast_to::<isize>()
        }

        // Discrete NN kernels: decode encoded neighbor IDs to ±1 one-hot steps.
        (KernelType::NearestNeighbor { .. }, DispCache::NN(nn)) => {
            nn.refresh(); // fills nn.vl with one-hot ±1 (isize)
            &nn.vl
        }

        // Mismatch protection (e.g., someone swapped kernel after construction)
        (kt, _) => panic!("Kernel/cache mismatch: got {kt:?}"),
    };

    // ---------------------- (3) Compose -------------------------
    // NOTE: `&*source_coords` coerces `&mut VectorList<_>` to `&VectorList<_>`.
    let sum = &*source_coords + displacements;
    *target_coords = sum;
}

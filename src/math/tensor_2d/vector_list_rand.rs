// src/math_foundations/vector_list_rand.rs
/*!
    Random fillers & utilities that work **in-place** on `VectorList<T>`.

    This file defines a small trait `VectorListRand` (requires `new` and `refresh`)
    and two implementers:
        - `HaarVectors`: unit vectors sampled uniformly on `S^{D-1}` (Haar).
        - `NNVectors`: nearest-neighbor one-hot ±1 direction vectors.

    Notes
    -----
        - It uses a DENSE backend by default!
        - `HaarVectors::refresh` draws i.i.d. Normal(0,1) and L2-normalizes each vector.
        - `NNVectors::refresh` samples integer **codes** in `[0, 2D)` and decodes them:
            axis = code / 2, sign = +1 if even else -1. It first zeroes all entries,
            then sets the one-hot signed axis per column.

    Both implement `VectorListRand`:
        let mut hv = HaarVectors::new(dim, n);
        hv.refresh();

        let mut nn = NNVectors::new(dim, n);
        nn.refresh();
*/
use rayon::prelude::*;

use crate::math::tensor::{
    dense::Tensor,
    dense_rand::{RandType, TensorRandFiller},
    tensor_trait::TensorTrait,
};

use crate::math::tensor_2d::vector_list::prelude::*;


// ============================================================================
// ------------------------------- Common Trait -------------------------------
// ============================================================================

/// Minimal interface for random vector-list generators.
pub trait VectorListRand {
    type Elem;

    /// Allocate an empty `[dim, n]` `VectorList` and any internal buffers.
    fn new(dim: usize, n: usize, num_rngs: Option<usize>) -> Self where Self: Sized;

    /// Refill the internal `VectorList` in-place, keeping shape `[dim, n]`.
    fn refresh(&mut self);
}





// ============================================================================
// -------------------------- Haar-random unit vectors ------------------------
// ============================================================================
#[derive(Debug, Clone)]
pub struct HaarVectors {
    pub vl: VectorList<f64>,
    pub dim: usize,
    pub n: usize,
    filler: TensorRandFiller, // RandType::Normal
}

impl VectorListRand for HaarVectors {
    type Elem = f64;

    fn new(dim: usize, n: usize, num_rngs: Option<usize>) -> Self {
        assert!(dim > 0, "HaarVectors::new: dim must be > 0");
        assert!(n > 0, "HaarVectors::new: n must be > 0");

        let vl = VectorList::<f64>::empty(dim, n, BackendKind::Dense);
        let filler = TensorRandFiller::new(
            RandType::Normal { mean: 0.0, std: 1.0 },
            num_rngs,
        );
        Self { vl, dim, n, filler }
    }

    #[inline]
    fn refresh(&mut self) {
        // 1) i.i.d. Gaussian entries on internal flat storage
        self.filler.refresh(&mut self.vl.as_tensor_mut());

        // 2) in-place L2 normalization per vector
        self.vl.normalize();
    }
}




// ============================================================================
// -------------------- Nearest-Neighbor one-hot ±1 vectors -------------------
// ============================================================================

#[derive(Debug, Clone)]
pub struct NNVectors {
    pub vl: VectorList<isize>,  // shape [dim, n], entries in {-1, 0, +1}
    pub dim: usize,
    pub n: usize,
    code_buf: Tensor<usize>,     // shape [n], holds codes in [0, 2*dim)
    code_filler: TensorRandFiller, // RandType::UniformInt over code range
}

impl VectorListRand for NNVectors {
    type Elem = isize;

    fn new(dim: usize, n: usize, num_rngs: Option<usize>) -> Self {
        assert!(dim > 0, "NNVectors::new: dim must be > 0");
        assert!(n > 0, "NNVectors::new: n must be > 0");

        // vector list storage
        let vl = VectorList::<isize>::empty(dim, n, BackendKind::Dense);

        // codes in [0, 2*dim)
        let code_buf = Tensor::<usize>::empty(vec![n].as_slice());

        let code_filler = TensorRandFiller::new(
            RandType::UniformInt { low: 0, high: (2 * dim) as i64 - 1 },
            num_rngs,
        );

        Self { vl, dim, n, code_buf, code_filler }
    }

    /// Randomize codes and decode into one-hot ±1 vectors.
    #[inline]
    fn refresh(&mut self) {
        // 1) sample codes in [0, 2*dim)
        self.code_filler.refresh(&mut self.code_buf);

        let n   = self.n;
        let codes: &[usize] = &self.code_buf.data;

        // 2) rewrite the whole [dim, n] matrix in parallel, row by row
        // row-major: each row is a contiguous chunk of length n
        self.vl.as_tensor_mut()
            .data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(axis, row)| {
                let axis = axis as usize;
                for (i, x) in row.iter_mut().enumerate() {
                    let code = codes[i];
                    let a    = code / 2;
                    let sign = if code % 2 == 0 { 1isize } else { -1isize };
                    *x = if a == axis { sign } else { 0 };
                }
            });
    }
}

// src/math_foundations/vector_list_rand.rs
/*!
Random fillers & utilities that work **in-place** on `VectorList<T>`.

This file defines a small trait `VectorListRand` (requires `new` and `refresh`)
and two implementers:
- `HaarVectors`: unit vectors sampled uniformly on `S^{D-1}` (Haar).
- `NNVectors`: nearest-neighbor one-hot ±1 direction vectors.

Notes
-----
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
use crate::math::tensor_2d::vector_list::VectorList;

// ============================================================================
// ------------------------------- Common Trait -------------------------------
// ============================================================================

/// Minimal interface for random vector-list generators.
pub trait VectorListRand {
    type Elem;

    /// Allocate an empty `[dim, n]` `VectorList` and any internal buffers.
    fn new(dim: usize, n: usize) -> Self where Self: Sized;

    /// Refill the internal `VectorList` in-place, keeping shape `[dim, n]`.
    fn refresh(&mut self);
}





// ============================================================================
// -------------------------- Haar-random unit vectors ------------------------
// ============================================================================

const CHUNK_SIZE: usize = 4096; // used to pick a default RNG count

#[derive(Debug, Clone)]
pub struct HaarVectors {
    pub vl: VectorList<f64>,
    pub dim: usize,
    pub n: usize,
    filler: TensorRandFiller, // RandType::Normal
}

impl VectorListRand for HaarVectors {
    type Elem = f64;

    fn new(dim: usize, n: usize) -> Self {
        assert!(dim > 0, "HaarVectors::new: dim must be > 0");
        assert!(n > 0, "HaarVectors::new: n must be > 0");

        let vl = VectorList::<f64>::empty(dim, n);
        let num_rngs = (dim * n) / CHUNK_SIZE + 1;

        let filler = TensorRandFiller::new(
            RandType::Normal { mean: 0.0, std: 1.0 },
            Some(num_rngs),
        );

        Self { vl, dim, n, filler }
    }

    #[inline]
    fn refresh(&mut self) {
        // 1) i.i.d. Gaussian entries on internal flat storage
        self.filler.refresh(&mut self.vl.matrix.tensor);

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

    fn new(dim: usize, n: usize) -> Self {
        assert!(dim > 0, "NNVectors::new: dim must be > 0");
        assert!(n > 0, "NNVectors::new: n must be > 0");

        // vector list storage
        let vl = VectorList::<isize>::empty(dim, n);

        // codes in [0, 2*dim)
        let code_buf = Tensor::<usize>::empty(vec![n].as_slice());
        // choose RNG count roughly proportional to n (these are 1D buffers)
        let num_rngs = (n / CHUNK_SIZE).max(1);

        let code_filler = TensorRandFiller::new(
            RandType::UniformInt { low: 0, high: (2 * dim) as i64 - 1 },
            Some(num_rngs),
        );

        Self { vl, dim, n, code_buf, code_filler }
    }

    /// Randomize codes and decode into one-hot ±1 vectors.
    #[inline]
    fn refresh(&mut self) {
        // Sample codes uniformly in [0, 2*dim)
        self.code_filler.refresh(&mut self.code_buf);

        // Zero the entire [dim, n] matrix first (parallel).
        self.vl.matrix.tensor
            .data
            .par_iter_mut()
            .for_each(|x| *x = 0isize);

        // Decode codes into one-hot ±1 per column.
        // axis = code / 2; sign = +1 if even else -1.
        for (i, &code) in self.code_buf.data.iter().enumerate() {
            let axis = code / 2;
            debug_assert!(axis < self.dim, "decoded axis out of range");
            let sign: isize = if code % 2 == 0 { 1 } else { -1 };
            self.vl.set(i as isize, axis as isize, sign);
        }
    }
}

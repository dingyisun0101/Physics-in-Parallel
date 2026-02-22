// src/math/tensor/rank_2/vector_list/rand.rs
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
use ndarray::Array2;

use crate::math::tensor::core::{
    dense::Tensor,
    dense_rand::{RandType, TensorRandFiller},
    tensor_trait::TensorTrait,
};

use crate::math::ndarray_convert::NdarrayConvert;
use super::VectorList;


// ============================================================================
// ------------------------------- Common Trait -------------------------------
// ============================================================================

/// Minimal interface for random vector-list generators.
pub trait VectorListRand {
    type Elem;

    /// Allocate an empty `[dim, n]` `VectorList` and any internal buffers.
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `dim` (`usize`): Parameter of type `usize` used by `new`.
    ///   - `n` (`usize`): Parameter of type `usize` used by `new`.
    ///   - `num_rngs` (`Option<usize>`): Parameter of type `Option<usize>` used by `new`.
    fn new(dim: usize, n: usize, num_rngs: Option<usize>) -> Self where Self: Sized;

    /// Refill the internal `VectorList` in-place, keeping shape `[dim, n]`.
    /// Annotation:
    /// - Purpose: Executes `refresh` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
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

    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `dim` (`usize`): Parameter of type `usize` used by `new`.
    ///   - `n` (`usize`): Parameter of type `usize` used by `new`.
    ///   - `num_rngs` (`Option<usize>`): Parameter of type `Option<usize>` used by `new`.
    fn new(dim: usize, n: usize, num_rngs: Option<usize>) -> Self {
        assert!(dim > 0, "HaarVectors::new: dim must be > 0");
        assert!(n > 0, "HaarVectors::new: n must be > 0");

        let vl = VectorList::<f64>::empty(dim, n);
        let filler = TensorRandFiller::new(
            RandType::Normal { mean: 0.0, std: 1.0 },
            num_rngs,
        );
        Self { vl, dim, n, filler }
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `refresh` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn refresh(&mut self) {
        // 1) i.i.d. Gaussian entries on internal flat storage
        self.filler.refresh(&mut self.vl.as_tensor_mut());

        // 2) in-place L2 normalization per vector
        self.vl.normalize();
    }
}

impl HaarVectors {
    /// Build a HaarVectors container from existing ndarray data (`[dim, n]`).
    ///
    /// The random filler is initialized to the standard Haar refresh distribution
    /// (`Normal { mean: 0, std: 1 }`), so a subsequent `refresh()` remains valid.
    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `ndarray` input.
    /// - Parameters:
    ///   - `array` (`&Array2<f64>`): ndarray input used for conversion/interoperability.
    pub fn from_ndarray(array: &Array2<f64>) -> Self {
        let vl = VectorList::<f64>::from_ndarray(array);
        let dim = vl.dim();
        let n = vl.num_vectors();
        let filler = TensorRandFiller::new(
            RandType::Normal { mean: 0.0, std: 1.0 },
            None,
        );
        Self { vl, dim, n, filler }
    }

    /// Export inner vector-list storage to ndarray with shape `[dim, n]`.
    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_ndarray(&self) -> Array2<f64> {
        self.vl.to_ndarray()
    }
}

impl NdarrayConvert for HaarVectors {
    type NdArray = Array2<f64>;

    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `ndarray` input.
    /// - Parameters:
    ///   - `array` (`&Self::NdArray`): ndarray input used for conversion/interoperability.
    fn from_ndarray(array: &Self::NdArray) -> Self {
        HaarVectors::from_ndarray(array)
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn to_ndarray(&self) -> Self::NdArray {
        HaarVectors::to_ndarray(self)
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

    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `dim` (`usize`): Parameter of type `usize` used by `new`.
    ///   - `n` (`usize`): Parameter of type `usize` used by `new`.
    ///   - `num_rngs` (`Option<usize>`): Parameter of type `Option<usize>` used by `new`.
    fn new(dim: usize, n: usize, num_rngs: Option<usize>) -> Self {
        assert!(dim > 0, "NNVectors::new: dim must be > 0");
        assert!(n > 0, "NNVectors::new: n must be > 0");

        // vector list storage
        let vl = VectorList::<isize>::empty(dim, n);

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
    /// Annotation:
    /// - Purpose: Executes `refresh` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn refresh(&mut self) {
        // 1) sample codes in [0, 2*dim)
        self.code_filler.refresh(&mut self.code_buf);

        let codes: &[usize] = &self.code_buf.data;

        // 2) rewrite all vectors in parallel.
        // VectorList stores physical shape [n, dim], so each vector is one contiguous row.
        self.vl.as_tensor_mut()
            .data
            .par_chunks_mut(self.dim)
            .enumerate()
            .for_each(|(i, row)| {
                let code = codes[i];
                for (axis, x) in row.iter_mut().enumerate() {
                    let a    = code / 2;
                    let sign = if code % 2 == 0 { 1isize } else { -1isize };
                    *x = if a == axis { sign } else { 0 };
                }
            });
    }
}

impl NNVectors {
    /// Build an NNVectors container from existing ndarray data (`[dim, n]`).
    ///
    /// The random code buffer/filler are initialized with defaults so `refresh()` remains valid.
    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `ndarray` input.
    /// - Parameters:
    ///   - `array` (`&Array2<isize>`): ndarray input used for conversion/interoperability.
    pub fn from_ndarray(array: &Array2<isize>) -> Self {
        let vl = VectorList::<isize>::from_ndarray(array);
        let dim = vl.dim();
        let n = vl.num_vectors();
        let code_buf = Tensor::<usize>::empty(vec![n].as_slice());
        let code_filler = TensorRandFiller::new(
            RandType::UniformInt { low: 0, high: (2 * dim) as i64 - 1 },
            None,
        );
        Self { vl, dim, n, code_buf, code_filler }
    }

    /// Export inner vector-list storage to ndarray with shape `[dim, n]`.
    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_ndarray(&self) -> Array2<isize> {
        self.vl.to_ndarray()
    }
}

impl NdarrayConvert for NNVectors {
    type NdArray = Array2<isize>;

    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `ndarray` input.
    /// - Parameters:
    ///   - `array` (`&Self::NdArray`): ndarray input used for conversion/interoperability.
    fn from_ndarray(array: &Self::NdArray) -> Self {
        NNVectors::from_ndarray(array)
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn to_ndarray(&self) -> Self::NdArray {
        NNVectors::to_ndarray(self)
    }
}

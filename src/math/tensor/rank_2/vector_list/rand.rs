/*!
Random vector-list generators.

Purpose:
    This module provides reusable random fillers for `VectorList` values while
    keeping random-number generation inside the rank-N tensor random
    infrastructure. The generators own their `VectorList` output and call
    `TensorRandFiller` to refresh dense buffers in-place.

Design:
    - `HaarVectors`:
        Fills the dense `[n, dim]` buffer with independent standard-normal
        values using `TensorRandFiller`, then asks `VectorList` to normalize
        each row. Each retrieved row is one Haar-distributed unit vector in
        `dim` dimensions.
    - `NNVectors`:
        Fills a rank-N dense code tensor with integer direction codes using
        `TensorRandFiller`, then decodes those codes through the vector-level
        row API. The stored rows are one-hot nearest-neighbor directions.
*/
use ndarray::Array2;
use serde_json::Value;

use crate::math::tensor::{RandType, TensorRandFiller, TensorTrait, dense::Tensor};

use super::VectorList;
use crate::math::ndarray_convert::NdarrayConvert;

// ============================================================================
// ------------------------------- Common Trait -------------------------------
// ============================================================================

/// Minimal interface for random vector-list generators.
pub trait VectorListRand {
    type Elem;

    /// Allocate output storage and rank-N random-fill buffers.
    fn new(dim: usize, n: usize, num_rngs: Option<usize>) -> Self
    where
        Self: Sized;

    /// Refill the internal `VectorList` in-place, keeping shape `[n, dim]`.
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

    /// Allocate a vector list and a rank-N normal random filler.
    fn new(dim: usize, n: usize, num_rngs: Option<usize>) -> Self {
        assert!(dim > 0, "HaarVectors::new: dim must be > 0");
        assert!(n > 0, "HaarVectors::new: n must be > 0");

        let vl = VectorList::<f64>::empty(dim, n);
        let filler = TensorRandFiller::new(
            RandType::Normal {
                mean: 0.0,
                std: 1.0,
            },
            num_rngs,
        );
        Self { vl, dim, n, filler }
    }

    #[inline]
    /// Refresh the dense buffer with rank-N normal random values and normalize rows.
    fn refresh(&mut self) {
        self.filler.refresh(self.vl.as_tensor_mut());
        self.vl.normalize();
    }
}

impl HaarVectors {
    /// Build a Haar generator around existing `[n, dim]` vector-list data.
    ///
    /// The random filler is initialized with the standard-normal distribution
    /// used by `refresh`, so callers can import data and later resume random
    /// Haar generation with the same object.
    #[inline]
    pub fn from_ndarray(array: &Array2<f64>) -> Self {
        let vl = VectorList::<f64>::from_ndarray(array);
        let dim = vl.dim();
        let n = vl.num_vectors();
        let filler = TensorRandFiller::new(
            RandType::Normal {
                mean: 0.0,
                std: 1.0,
            },
            None,
        );
        Self { vl, dim, n, filler }
    }

    /// Export inner vector-list storage to ndarray with shape `[n, dim]`.
    #[inline]
    pub fn to_ndarray(&self) -> Array2<f64> {
        self.vl.to_ndarray()
    }

    #[inline]
    /// Convert this Haar vector batch into a structured JSON value.
    pub fn serialize_value(&self) -> Result<Value, serde_json::Error> {
        self.vl.serialize_value()
    }

    #[inline]
    /// Convert this Haar vector batch into pretty JSON text.
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.vl.serialize()
    }
}

impl NdarrayConvert for HaarVectors {
    type NdArray = Array2<f64>;

    #[inline]
    fn from_ndarray(array: &Self::NdArray) -> Self {
        HaarVectors::from_ndarray(array)
    }

    #[inline]
    fn to_ndarray(&self) -> Self::NdArray {
        HaarVectors::to_ndarray(self)
    }
}

// ============================================================================
// -------------------- Nearest-Neighbor one-hot ±1 vectors -------------------
// ============================================================================

#[derive(Debug, Clone)]
pub struct NNVectors {
    pub vl: VectorList<isize>, // shape [n, dim], entries in {-1, 0, +1}
    pub dim: usize,
    pub n: usize,
    code_buf: Tensor<usize>,       // shape [n], holds codes in [0, 2*dim)
    code_filler: TensorRandFiller, // RandType::UniformInt over code range
}

impl VectorListRand for NNVectors {
    type Elem = isize;

    /// Allocate output rows and a rank-N integer-code random filler.
    fn new(dim: usize, n: usize, num_rngs: Option<usize>) -> Self {
        assert!(dim > 0, "NNVectors::new: dim must be > 0");
        assert!(n > 0, "NNVectors::new: n must be > 0");

        let vl = VectorList::<isize>::empty(dim, n);
        let code_buf = Tensor::<usize>::empty(vec![n].as_slice());

        let code_filler = TensorRandFiller::new(
            RandType::UniformInt {
                low: 0,
                high: (2 * dim) as i64 - 1,
            },
            num_rngs,
        );

        Self {
            vl,
            dim,
            n,
            code_buf,
            code_filler,
        }
    }

    #[inline]
    /// Refresh rank-N integer codes and decode them into vector-list rows.
    fn refresh(&mut self) {
        self.code_filler.refresh(&mut self.code_buf);

        self.vl.par_for_each_vec_mut(|i, row| {
            decode_nearest_neighbor_code(self.code_buf.get(&[i as isize]), row);
        });
    }
}

#[inline]
fn decode_nearest_neighbor_code(code: usize, row: &mut [isize]) {
    let axis = code / 2;
    let sign = if code % 2 == 0 { 1isize } else { -1isize };
    for (k, x) in row.iter_mut().enumerate() {
        *x = if k == axis { sign } else { 0 };
    }
}

impl NNVectors {
    /// Build a nearest-neighbor generator around existing `[n, dim]` rows.
    ///
    /// The integer-code filler is initialized with the same direction-code
    /// range used by `refresh`, so imported data can later be replaced by new
    /// random nearest-neighbor directions.
    #[inline]
    pub fn from_ndarray(array: &Array2<isize>) -> Self {
        let vl = VectorList::<isize>::from_ndarray(array);
        let dim = vl.dim();
        let n = vl.num_vectors();
        let code_buf = Tensor::<usize>::empty(vec![n].as_slice());
        let code_filler = TensorRandFiller::new(
            RandType::UniformInt {
                low: 0,
                high: (2 * dim) as i64 - 1,
            },
            None,
        );
        Self {
            vl,
            dim,
            n,
            code_buf,
            code_filler,
        }
    }

    /// Export inner vector-list storage to ndarray with shape `[n, dim]`.
    #[inline]
    pub fn to_ndarray(&self) -> Array2<isize> {
        self.vl.to_ndarray()
    }

    #[inline]
    /// Convert this nearest-neighbor vector batch into a structured JSON value.
    pub fn serialize_value(&self) -> Result<Value, serde_json::Error> {
        self.vl.serialize_value()
    }

    #[inline]
    /// Convert this nearest-neighbor vector batch into pretty JSON text.
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.vl.serialize()
    }
}

impl NdarrayConvert for NNVectors {
    type NdArray = Array2<isize>;

    #[inline]
    fn from_ndarray(array: &Self::NdArray) -> Self {
        NNVectors::from_ndarray(array)
    }

    #[inline]
    fn to_ndarray(&self) -> Self::NdArray {
        NNVectors::to_ndarray(self)
    }
}

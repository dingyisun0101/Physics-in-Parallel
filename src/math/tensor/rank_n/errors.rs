//! Shared tensor error types.
//!
//! These errors describe tensor-level contract violations independently from
//! any dense or sparse storage implementation.

use crate::math::scalar::ScalarCastError;

/// Standard result type for fallible tensor operations.
pub type TensorResult<T> = Result<T, TensorError>;

/// Error returned by fallible tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Shape contains no axes or contains at least one zero-length axis.
    InvalidShape { shape: Vec<usize> },
    /// Shape product overflowed `usize`.
    ShapeProductOverflow { shape: Vec<usize> },
    /// Two tensors were expected to have the same shape.
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },
    /// An index has the wrong rank for the tensor shape.
    RankMismatch {
        shape: Vec<usize>,
        index_rank: usize,
    },
    /// Scalar conversion failed while casting tensor elements.
    ScalarCast(ScalarCastError),
}

impl From<ScalarCastError> for TensorError {
    #[inline]
    fn from(value: ScalarCastError) -> Self {
        Self::ScalarCast(value)
    }
}

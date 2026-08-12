//! Shared tensor error types.
//!
//! These errors describe tensor-level contract violations independently from
//! any dense or sparse storage implementation.

use core::fmt;

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
    /// An operation expected a particular tensor rank.
    ExpectedRank {
        operation: &'static str,
        expected: usize,
        actual: usize,
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

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape { shape } => {
                write!(
                    f,
                    "tensor shape must contain at least one nonzero axis; got {shape:?}"
                )
            }
            Self::ShapeProductOverflow { shape } => {
                write!(f, "tensor shape product overflowed usize; got {shape:?}")
            }
            Self::ShapeMismatch { lhs, rhs } => {
                write!(f, "tensor shape mismatch: lhs={lhs:?}, rhs={rhs:?}")
            }
            Self::RankMismatch { shape, index_rank } => {
                write!(
                    f,
                    "tensor index rank mismatch: shape rank={}, index rank={index_rank}",
                    shape.len()
                )
            }
            Self::ExpectedRank {
                operation,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "{operation} requires rank {expected}, but tensor rank is {actual}"
                )
            }
            Self::ScalarCast(error) => write!(f, "tensor scalar cast failed: {error}"),
        }
    }
}

impl std::error::Error for TensorError {}

/// Validate that a tensor shape has at least one axis and no zero-length axes.
#[inline]
pub fn validate_shape(shape: &[usize]) -> TensorResult<()> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(TensorError::InvalidShape {
            shape: shape.to_vec(),
        });
    }
    Ok(())
}

/// Compute the dense logical size implied by `shape`.
#[inline]
pub fn checked_num_elements(shape: &[usize]) -> TensorResult<usize> {
    validate_shape(shape)?;

    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| TensorError::ShapeProductOverflow {
                shape: shape.to_vec(),
            })
    })
}

/// Validate that two tensors have the same shape.
#[inline]
pub fn ensure_same_shape(lhs: &[usize], rhs: &[usize]) -> TensorResult<()> {
    if lhs != rhs {
        return Err(TensorError::ShapeMismatch {
            lhs: lhs.to_vec(),
            rhs: rhs.to_vec(),
        });
    }
    Ok(())
}

/// Validate that an index has the same rank as a shape.
#[inline]
pub fn ensure_index_rank(shape: &[usize], index_rank: usize) -> TensorResult<()> {
    if shape.len() != index_rank {
        return Err(TensorError::RankMismatch {
            shape: shape.to_vec(),
            index_rank,
        });
    }
    Ok(())
}

/// Validate an operation-specific expected rank.
#[inline]
pub fn ensure_rank(operation: &'static str, actual: usize, expected: usize) -> TensorResult<()> {
    if actual != expected {
        return Err(TensorError::ExpectedRank {
            operation,
            expected,
            actual,
        });
    }
    Ok(())
}

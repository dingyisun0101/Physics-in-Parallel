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
    /// An axis or the total linear domain cannot be represented by `isize`.
    IndexSpaceOverflow { shape: Vec<usize> },
    /// Two tensors were expected to have the same shape.
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },
    /// Dense row-major data did not match the shape's logical element count.
    DataLengthMismatch { expected: usize, actual: usize },
    /// Coordinates have the wrong rank for the tensor shape.
    RankMismatch {
        shape: Vec<usize>,
        coordinate_rank: usize,
    },
    /// A coordinate component is outside its axis extent.
    CoordinateOutOfBounds {
        axis: usize,
        coordinate: usize,
        extent: usize,
    },
    /// Sparse construction specified one logical coordinate more than once.
    DuplicateCoordinate { coordinates: Vec<usize> },
    /// An operation expected a particular tensor rank.
    ExpectedRank {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Two axes that must agree for an operation have different extents.
    DimensionMismatch {
        operation: &'static str,
        lhs: usize,
        rhs: usize,
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
            Self::IndexSpaceOverflow { shape } => write!(
                f,
                "tensor shape exceeds the signed public index space; got {shape:?}"
            ),
            Self::ShapeMismatch { lhs, rhs } => {
                write!(f, "tensor shape mismatch: lhs={lhs:?}, rhs={rhs:?}")
            }
            Self::DataLengthMismatch { expected, actual } => write!(
                f,
                "tensor data length mismatch: expected {expected}, got {actual}"
            ),
            Self::RankMismatch {
                shape,
                coordinate_rank,
            } => {
                write!(
                    f,
                    "tensor coordinate rank mismatch: shape rank={}, coordinate rank={coordinate_rank}",
                    shape.len()
                )
            }
            Self::CoordinateOutOfBounds {
                axis,
                coordinate,
                extent,
            } => write!(
                f,
                "tensor coordinate {coordinate} is outside axis {axis} with extent {extent}"
            ),
            Self::DuplicateCoordinate { coordinates } => {
                write!(
                    f,
                    "tensor coordinate {coordinates:?} appears more than once"
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
            Self::DimensionMismatch {
                operation,
                lhs,
                rhs,
            } => write!(
                f,
                "{operation} dimension mismatch: left extent {lhs}, right extent {rhs}"
            ),
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

    if shape
        .iter()
        .any(|&dimension| dimension > isize::MAX as usize)
    {
        return Err(TensorError::IndexSpaceOverflow {
            shape: shape.to_vec(),
        });
    }

    let size = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| TensorError::ShapeProductOverflow {
                shape: shape.to_vec(),
            })
    })?;
    if size > isize::MAX as usize {
        return Err(TensorError::IndexSpaceOverflow {
            shape: shape.to_vec(),
        });
    }
    Ok(size)
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

/// Validate that coordinates have the same rank as a shape.
#[inline]
pub fn ensure_coordinate_rank(shape: &[usize], coordinate_rank: usize) -> TensorResult<()> {
    if shape.len() != coordinate_rank {
        return Err(TensorError::RankMismatch {
            shape: shape.to_vec(),
            coordinate_rank,
        });
    }
    Ok(())
}

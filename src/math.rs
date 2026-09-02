/*!
Core math foundations.
*/

pub(crate) mod io;
pub(crate) mod scalar;
pub(crate) mod tensor;

// Minimal foundational facade. Backend and interchange types live in
// `crate::advanced`.
pub use scalar::{Complex, Scalar, ScalarCastError};
pub use tensor::{
    Matrix, MatrixError, RandType, StorageKind, Tensor, TensorError, TensorRandError,
    TensorRandFiller, Values, VectorList, VectorListError,
};

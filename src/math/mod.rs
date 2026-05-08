/*!
Core math foundations.
*/

pub mod io;
pub mod prelude;
pub mod scalar;
pub mod tensor;

// Canonical top-level exports.
pub use io::ndarray::NdarrayConvert;
pub use scalar::{Complex, Scalar, ScalarCastError, ScalarSerde};
pub use tensor::{
    Backend, Dense, RandType, RngKind, Sparse, Tensor, TensorError, TensorRandError,
    TensorRandFiller, TensorResult, TensorTrait,
};

/*!
Core math foundations.
*/

pub mod scalar;
pub mod tensor;
pub mod ndarray_convert;
pub mod prelude;

// Canonical top-level exports.
pub use crate::io::ndarray::NdarrayConvert;
pub use scalar::{Scalar, ScalarSerde};
pub use tensor::{Backend, Dense, Sparse, Tensor};

pub(crate) mod rank_2;
pub(crate) mod rank_n;

pub use rank_2::matrix::{DenseMatrix, Matrix, MatrixError};
pub use rank_2::vector_list::VectorList;
pub use rank_n::dense_rand::{RandType, TensorRandError, TensorRandFiller};
pub use rank_n::{Tensor, TensorError};

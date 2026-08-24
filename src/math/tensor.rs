pub(crate) mod rank_2;
pub(crate) mod rank_n;

pub use rank_2::matrix::{DenseMatrix, Matrix, MatrixError};
pub use rank_2::vector_list::VectorList;
pub use rank_n::dense_rand::{
    DEFAULT_RANDOM_MAX_THREADS, RandType, TensorRandError, TensorRandFiller,
};
pub use rank_n::{Tensor, TensorError};

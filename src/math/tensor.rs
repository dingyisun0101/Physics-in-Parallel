pub(crate) mod rank_2;
pub(crate) mod rank_n;
mod universal;
mod universal_matrix;
mod universal_vector_list;

pub use rank_n::TensorError;
pub use rank_n::dense_rand::{RandType, TensorRandError, TensorRandFiller};
pub use universal::{StorageKind, Tensor, Values};
pub use universal_matrix::{Matrix, MatrixError};
pub use universal_vector_list::{VectorList, VectorListError};

pub mod rank_2;
pub mod rank_n;

pub use rank_2::{
    HaarVectors, Matrix, MatrixTrait, NNVectors, Tensor2D, VectorList, VectorListRand,
};
pub use rank_n::dense_rand::{RandType, RngKind, TensorRandError, TensorRandFiller};
pub use rank_n::tensor_trait::TensorTrait;
pub use rank_n::{
    Backend, Dense, Sparse, Tensor, TensorError, TensorResult, dense, dense_rand, errors, sparse,
    tensor_trait,
};

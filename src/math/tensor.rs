pub mod rank_2;
pub mod rank_n;

pub use rank_2::{
    AntiSymmetricMatrix, DenseMatrix, DiagonalMatrix, HaarVectors, LowerTriangularMatrix, Matrix,
    MatrixBackend, MatrixError, NNVectors, SparseMatrix, StrictLowerTriangularMatrix,
    StrictUpperTriangularMatrix, SymmetricMatrix, UpperTriangularMatrix, VectorList,
    VectorListRand,
};
pub use rank_n::dense_rand::{RandType, TensorRandError, TensorRandFiller};
pub use rank_n::tensor_trait::TensorTrait;
pub use rank_n::{
    Backend, Dense, RowMajorLayout, Sparse, Tensor, TensorError, TensorResult, dense, dense_rand,
    errors, flat_index_wrapped, layout, sparse, tensor_trait,
};

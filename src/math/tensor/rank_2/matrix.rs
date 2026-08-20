pub(crate) mod generic;
pub(crate) mod matrix_backend_trait;
pub(crate) mod structured;

pub use generic::{DenseMatrix, Matrix, MatrixError, RankNDense, RankNSparse, SparseMatrix};
pub use matrix_backend_trait::MatrixBackend;
pub use structured::{
    AntiSymmetricMatrix, DiagonalMatrix, LowerTriangularMatrix, StrictLowerTriangularMatrix,
    StrictUpperTriangularMatrix, SymmetricMatrix, UpperTriangularMatrix,
};

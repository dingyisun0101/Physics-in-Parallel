pub mod generic;
pub mod matrix_backend_trait;
pub mod structured;

pub use generic::{DenseMatrix, Matrix, RankNDense, RankNSparse, SparseMatrix};
pub use matrix_backend_trait::MatrixBackend;
pub use structured::{
    AntiSymmetric, AntiSymmetricMatrix, Diagonal, DiagonalMatrix, LowerTriangularMatrix,
    StrictLowerTriangularMatrix, StrictUpperTriangularMatrix, Symmetric, SymmetricMatrix,
    Triangular, UpperTriangularMatrix,
};

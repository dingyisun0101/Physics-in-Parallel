/*!
User-facing prelude for the `math` module.

Import with:
`use physics_in_parallel::math::prelude::*;`
*/

// Scalar traits
pub use crate::math::{Complex, Scalar, ScalarCastError, ScalarSerde};

// Common conversion trait
pub use crate::math::NdarrayConvert;

// Generic tensor front API
pub use crate::math::tensor::{
    Backend, Dense as DenseBackend, RandType, Sparse as SparseBackend, Tensor, TensorError,
    TensorRandError, TensorRandFiller, TensorResult, TensorTrait, dense, dense_rand, sparse,
    tensor_trait,
};
pub use crate::rng::{IndexedRng, RngConfig, RngConfigError, RngMethod};

// Rank-2 tensor and matrix/vector abstractions
pub use crate::math::tensor::rank_2::{
    AntiSymmetricMatrix, DenseMatrix, DiagonalMatrix, HaarVectors, LowerTriangularMatrix, Matrix,
    MatrixBackend, NNVectors, SparseMatrix, StrictLowerTriangularMatrix,
    StrictUpperTriangularMatrix, SymmetricMatrix, UpperTriangularMatrix, VectorList,
    VectorListRand,
};

/*!
User-facing prelude for the `math` module.

Import with:
`use physics_in_parallel::math::prelude::*;`
*/

// Scalar traits
pub use crate::math::{Scalar, ScalarSerde};

// Common conversion trait
pub use crate::math::NdarrayConvert;

// Unified tensor front API
pub use crate::math::tensor::{Backend, Dense as DenseBackend, Sparse as SparseBackend, Tensor};

// Tensor utility traits/types
pub use crate::math::tensor::core::dense_rand::{RandType, TensorRandFiller};
pub use crate::math::tensor::core::tensor_trait::TensorTrait;

// Rank-2 tensor and matrix/vector abstractions
pub use crate::math::tensor::rank_2::{HaarVectors, Matrix, MatrixTrait, NNVectors, Tensor2D, VectorList, VectorListRand};

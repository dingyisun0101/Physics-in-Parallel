/*!
User-facing prelude for the `math` module.

Import with:
`use physics_in_parallel::math::prelude::*;`
*/

// Scalar traits
pub use crate::math::scalar::{Scalar, ScalarSerde};

// Common conversion trait
pub use crate::math::ndarray_convert::NdarrayConvert;

// Unified tensor front API
pub use crate::math::tensor::core::{
    Backend,
    Dense as DenseBackend,
    Sparse as SparseBackend,
    Tensor,
};

// Tensor utility traits/types
pub use crate::math::tensor::core::dense_rand::{RandType, TensorRandFiller};
pub use crate::math::tensor::core::tensor_trait::TensorTrait;

// Rank-2 tensor and matrix/vector abstractions
pub use crate::math::tensor::rank_2::dense::Tensor2D;
pub use crate::math::tensor::rank_2::matrix::dense::Matrix;
pub use crate::math::tensor::rank_2::matrix::matrix_trait::MatrixTrait;
pub use crate::math::tensor::rank_2::vector_list::VectorList;
pub use crate::math::tensor::rank_2::vector_list::rand::{
    HaarVectors,
    NNVectors,
    VectorListRand,
};


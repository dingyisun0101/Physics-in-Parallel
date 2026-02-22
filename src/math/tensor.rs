pub mod core;
pub mod rank_2;

pub use core::{Backend, Dense, Sparse, Tensor};
pub use rank_2::{HaarVectors, Matrix, MatrixTrait, NNVectors, Tensor2D, VectorList, VectorListRand};

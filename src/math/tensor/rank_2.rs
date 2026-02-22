pub mod dense;
pub mod sparse;
pub mod matrix;
pub mod vector_list;
pub use vector_list::rand as vector_list_rand;

pub use dense::Tensor2D;
pub use matrix::{Matrix, MatrixTrait};
pub use vector_list::{HaarVectors, NNVectors, VectorList, VectorListRand};

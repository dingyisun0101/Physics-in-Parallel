pub mod dense;
pub mod sparse;
pub mod dense_rand;
pub mod tensor_trait;
pub mod unified;

pub use unified::{Backend, Dense, Sparse, Tensor};

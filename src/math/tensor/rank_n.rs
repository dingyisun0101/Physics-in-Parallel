pub mod dense;
pub mod dense_rand;
pub mod errors;
pub mod generic;
pub mod layout;
pub mod ops;
pub mod sparse;
pub mod tensor_trait;

pub use errors::{TensorError, TensorResult};
pub use generic::{Backend, Dense, Sparse, Tensor};
pub use layout::{RowMajorLayout, flat_index_wrapped};

pub(crate) mod dense;
pub(crate) mod dense_rand;
pub(crate) mod errors;
pub(crate) mod generic;
pub(crate) mod layout;
pub(crate) mod ops;
pub(crate) mod sparse;
pub(crate) mod tensor_trait;

pub use errors::{TensorError, TensorResult};
pub use generic::{Dense, Sparse, Tensor};
pub use layout::RowMajorLayout;

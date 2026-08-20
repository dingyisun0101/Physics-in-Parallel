#![allow(clippy::tabs_in_doc_comments)]

pub(crate) mod complex;
pub(crate) mod real;
pub(crate) mod scalar_trait;

pub use num_complex::Complex;
pub use scalar_trait::{Scalar, ScalarCastError, ScalarSerde};

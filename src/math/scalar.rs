#![allow(clippy::tabs_in_doc_comments)]

pub mod complex;
pub mod real;
pub mod scalar_trait;

pub use scalar_trait::{Scalar, ScalarCastError, ScalarSerde};

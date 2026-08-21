//! Generic sampling data structures independent of geometry and RNG ownership.

mod shuffle;
mod weighted;

pub(crate) use shuffle::shuffle_slice_indexed;
pub use weighted::{DynamicWeightedIndex, DynamicWeightedIndexError};

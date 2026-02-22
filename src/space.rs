/*!
Core space abstractions and utilities.
*/

pub mod space_trait;
pub mod discrete;
pub mod continuous;
pub mod kernel;
pub mod prelude;

// Canonical top-level exports.
pub use kernel::{Kernel, KernelType};
pub use space_trait::Space;

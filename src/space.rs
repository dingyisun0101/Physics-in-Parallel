/*!
Core space abstractions and utilities.
*/

pub mod continuous;
pub mod discrete;
pub mod kernel;
pub mod prelude;
pub mod space_trait;

// Canonical top-level exports.
pub use kernel::{Kernel, KernelType};
pub use space_trait::Space;

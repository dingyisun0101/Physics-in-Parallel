/*!
Core space abstractions and utilities.
*/

pub mod continuous;
pub mod discrete;
pub mod io;
pub mod prelude;
pub mod space_trait;

// Canonical top-level exports.
pub use discrete::square_lattice::{Kernel, KernelType};
pub use space_trait::Space;

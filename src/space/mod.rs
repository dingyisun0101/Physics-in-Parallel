/*!
Core space abstractions and utilities.
*/

pub mod continuous;
pub mod discrete;
pub mod io;
pub mod prelude;
pub mod space_trait;

// Canonical top-level exports.
pub use continuous::boundary::{ClampBox, ContinuousBoundary, PeriodicBox, ReflectBox};
pub use continuous::sampling::{VectorSamplingMethod, sample_vectors};
pub use discrete::square_lattice::{
    Kernel, KernelError, KernelType, PairGenerationError, PairRandomKey,
};
pub use space_trait::Space;

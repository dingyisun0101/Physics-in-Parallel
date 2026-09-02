/*!
Core space abstractions and utilities.
*/

pub mod continuous;
pub mod discrete;
pub(crate) mod io;

pub use continuous::boundary::{
    BoundaryError, ClampBox, ContinuousBoundary, PeriodicBox, ReflectBox,
};
pub use continuous::sampling::{VectorSamplingError, VectorSamplingMethod, sample_vectors};
pub use discrete::square_lattice::{
    BoundaryCondition, KernelError, KernelType, PairGenerationError, PairGenerator, PairingMethod,
    SourceMode, SquareLattice, SquareLatticeGeometry, SquareLatticeGeometryError,
    SquareLatticeInitMethod,
};

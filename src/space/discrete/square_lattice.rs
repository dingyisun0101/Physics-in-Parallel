/*!
Square-lattice discrete spaces.

Purpose:
    This module groups the data representation and lattice-specific random
    pairing and displacement tools for square, cubic, and hypercubic lattices. Keeping
    these files under `square_lattice` leaves room for other discrete-space
    families, such as triangular or graph-based spaces, to define their own
    representation and displacement semantics later.
*/

pub mod kernel;
pub mod pairing;
pub(crate) mod random;
pub mod representation;

#[allow(deprecated)]
pub use kernel::{
    Kernel, KernelError, KernelType, NearestNeighborKernel, PowerLawKernel, UniformDistanceKernel,
    UniformKernel, create_kernel, try_create_kernel,
};
#[allow(deprecated)]
pub use pairing::{
    PairGenerationError, PairGenerator, PairGeneratorConfig, PairingMethod, RandPairGenerator,
    SourceMode,
};
pub use representation::{
    BoundaryCondition, SquareLattice, SquareLatticeConfig, SquareLatticeConfigError,
    SquareLatticeInitMethod,
};

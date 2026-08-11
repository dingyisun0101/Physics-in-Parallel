/*!
Square-lattice discrete spaces.

Purpose:
    This module groups the data representation and lattice-specific random
    displacement tools for square, cubic, and hypercubic lattices. Keeping
    these files under `square_lattice` leaves room for other discrete-space
    families, such as triangular or graph-based spaces, to define their own
    representation and displacement semantics later.
*/

pub mod displacement;
pub mod kernel;
pub(crate) mod random;
pub mod representation;

pub use displacement::{PairGenerationError, RandPairGenerator, SourceMode};
pub use kernel::{
    Kernel, KernelError, KernelType, NearestNeighborKernel, PowerLawKernel, UniformKernel,
    create_kernel, try_create_kernel,
};
pub use representation::{
    BoundaryCondition, SquareLattice, SquareLatticeConfig, SquareLatticeConfigError,
    SquareLatticeInitMethod,
};

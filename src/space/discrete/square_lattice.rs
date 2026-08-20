/*!
Square-lattice discrete spaces.

Purpose:
    This module groups the data representation and lattice-specific random
    pairing and displacement tools for square, cubic, and hypercubic lattices. Keeping
    these files under `square_lattice` leaves room for other discrete-space
    families, such as triangular or graph-based spaces, to define their own
    representation and displacement semantics later.
*/

pub(crate) mod kernel;
pub(crate) mod pairing;
pub(crate) mod random;
pub(crate) mod representation;

pub use kernel::{KernelError, KernelType};
pub use pairing::{
    PairGenerationError, PairGenerator, PairGeneratorConfig, PairingMethod, SourceMode,
};
pub use representation::{
    BoundaryCondition, SquareLattice, SquareLatticeConfig, SquareLatticeConfigError,
    SquareLatticeInitMethod,
};

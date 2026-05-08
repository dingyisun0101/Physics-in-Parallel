/*!
User-facing prelude for the `space` module.

Import with:
`use physics_in_parallel::space::prelude::*;`
*/

pub use crate::space::space_trait::Space;

pub use crate::space::kernel::{
    Kernel, KernelType, NearestNeighborKernel, PowerLawKernel, UniformKernel, create_kernel,
};

pub use crate::space::discrete::square_lattice::{
    BoundaryCondition, SquareLattice, SquareLatticeConfig, SquareLatticeInitMethod, VacancyValue,
};

pub use crate::space::discrete::square_lattice::RandPairGenerator;
pub use crate::space::io::square_lattice::save_square_lattice;

/*!
User-facing prelude for the `space` module.

Import with:
`use physics_in_parallel::space::prelude::*;`
*/

pub use crate::space::space_trait::Space;

pub use crate::space::kernel::{
    Kernel, KernelType, NearestNeighborKernel, PowerLawKernel, UniformKernel, create_kernel,
};

pub use crate::space::discrete::representation::{
    Grid, GridConfig, GridInitMethod, VacancyValue, save_grid,
};

pub use crate::space::discrete::displacement::RandPairGenerator;

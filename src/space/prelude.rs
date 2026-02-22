/*!
User-facing prelude for the `space` module.

Import with:
`use physics_in_parallel::space::prelude::*;`
*/

pub use crate::space::space_trait::Space;

pub use crate::space::kernel::{
    create_kernel,
    Kernel,
    KernelType,
    NearestNeighborKernel,
    PowerLawKernel,
    UniformKernel,
};

pub use crate::space::discrete::representation::{
    save_grid,
    Grid,
    GridConfig,
    GridInitMethod,
    VacancyValue,
};

pub use crate::space::discrete::displacement::RandPairGenerator;

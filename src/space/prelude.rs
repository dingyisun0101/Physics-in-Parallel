/*!
User-facing prelude for the `space` module.

Import with:
`use physics_in_parallel::space::prelude::*;`
*/

pub use crate::space::space_trait::Space;

pub use crate::space::continuous::boundary::{
    BoundaryError, ClampBox, ContinuousBoundary, PeriodicBox, ReflectBox,
};
pub use crate::space::continuous::sampling::{
    VectorSamplingError, VectorSamplingMethod, sample_vectors,
};
pub use crate::space::discrete::square_lattice::{
    BoundaryCondition, Kernel, KernelError, KernelType, NearestNeighborKernel,
    PAIR_RANDOM_KEY_ENCODING, PAIR_RANDOM_METHOD, PAIR_RANDOM_VERSION, PairGenerationError,
    PairRandomKey, PowerLawKernel, RandPairGenerator, SourceMode, SquareLattice,
    SquareLatticeConfig, SquareLatticeInitMethod, UniformKernel, VacancyValue, create_kernel,
    try_create_kernel,
};
pub use crate::space::io::square_lattice::save_square_lattice;

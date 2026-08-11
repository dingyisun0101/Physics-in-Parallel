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
    BoundaryCondition, INDEXED_RANDOM_KEY_ENCODING, INDEXED_RANDOM_METHOD, INDEXED_RANDOM_VERSION,
    Kernel, KernelError, KernelType, NearestNeighborKernel, PairGenerationError, PowerLawKernel,
    RandPairGenerator, RandomKey, SourceMode, SquareLattice, SquareLatticeConfig,
    SquareLatticeConfigError, SquareLatticeInitMethod, UniformKernel, create_kernel,
    try_create_kernel,
};
pub use crate::space::io::square_lattice::save_square_lattice;

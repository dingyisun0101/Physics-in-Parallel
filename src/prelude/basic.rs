//! Foundational user API for mathematics, randomness, spaces, and pairing.

pub use crate::math::{
    Complex, Matrix, MatrixError, RandType, Scalar, ScalarCastError, StorageKind, Tensor,
    TensorError, TensorRandError, TensorRandFiller, Values, VectorList, VectorListError,
};
pub use crate::rng::{IndexedRng, ResolvedRng, RngError, RngMethod};
pub use crate::space::{
    BoundaryCondition, BoundaryError, ClampBox, ContinuousBoundary, KernelError, KernelType,
    PairGenerationError, PairGenerator, PairGeneratorConfig, PairingMethod, PeriodicBox,
    ReflectBox, SourceMode, SquareLattice, SquareLatticeConfig, SquareLatticeConfigError,
    SquareLatticeInitMethod, VectorSamplingError, VectorSamplingMethod, sample_vectors,
};
pub use crate::threading::{ParallelismError, max_threads, set_max_threads};

//! Foundational user API for mathematics, randomness, spaces, and pairing.

pub use crate::math::{
    Complex, DenseMatrix, Matrix, MatrixError, RandType, Scalar, ScalarCastError, Tensor,
    TensorError, TensorRandError, TensorRandFiller, VectorList,
};
pub use crate::parallel::{
    DEFAULT_PARALLEL_PARTITIONS, ParallelismError, parallel_partitions, set_parallel_partitions,
};
pub use crate::rng::{IndexedRng, RngConfig, RngConfigError, RngMethod};
pub use crate::space::{
    BoundaryCondition, BoundaryError, ClampBox, ContinuousBoundary, KernelError, KernelType,
    PairGenerationError, PairGenerator, PairGeneratorConfig, PairingMethod, PeriodicBox,
    ReflectBox, SourceMode, SquareLattice, SquareLatticeConfig, SquareLatticeConfigError,
    SquareLatticeInitMethod, VectorSamplingError, VectorSamplingMethod, sample_vectors,
};

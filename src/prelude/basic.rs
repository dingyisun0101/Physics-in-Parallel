//! Foundational user API for mathematics, randomness, spaces, and pairing.

pub use crate::math::{
    Backend, Complex, Matrix, MatrixBuilder, MatrixError, RandType, Scalar, ScalarCastError,
    Tensor, TensorBuilder, TensorError, TensorRandError, TensorRandFiller, Values, VectorList,
    VectorListBuilder, VectorListError,
};
pub use crate::rng::{IndexedRng, ResolvedRng, RngError, RngMethod};
pub use crate::space::{
    BoundaryCondition, BoundaryError, ClampBox, ContinuousBoundary, KernelError, KernelType,
    PairGenerationError, PairGenerator, PairingMethod, PeriodicBox, ReflectBox, SourceMode,
    SquareLattice, SquareLatticeGeometry, SquareLatticeGeometryError, SquareLatticeInitMethod,
    VectorSamplingError, VectorSamplingMethod, sample_vectors,
};
pub use crate::threading::{ParallelismError, max_threads, set_max_threads};

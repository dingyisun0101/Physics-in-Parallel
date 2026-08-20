pub mod boundary;
pub mod representation;
pub mod sampling;

pub use boundary::{BoundaryError, ClampBox, ContinuousBoundary, PeriodicBox, ReflectBox};
pub use sampling::{VectorSamplingError, VectorSamplingMethod, sample_vectors};

//! Thread-pool integration and adaptive work planning.
//!
//! PiP executes parallel work on the current Rayon pool. It does not create a
//! pool unless a caller explicitly constructs [`ComputePool`] or invokes
//! [`with_threads`].

use std::error::Error;
use std::fmt;

/// Minimum flat element count before adaptive elementwise work uses Rayon.
pub(crate) const MIN_PARALLEL_ELEMENTS: usize = 16_384;

/// Minimum estimated scalar operations before a cost-aware kernel uses Rayon.
pub(crate) const MIN_PARALLEL_OPERATIONS: usize = 131_072;

/// Failure while explicitly constructing a PiP convenience thread pool.
#[derive(Debug)]
#[non_exhaustive]
pub enum ComputePoolError {
    /// A pool must contain at least one worker.
    ZeroThreads,
    /// Rayon could not construct the requested pool.
    Build(rayon::ThreadPoolBuildError),
}

impl fmt::Display for ComputePoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroThreads => formatter.write_str("PiP compute-pool threads must be positive"),
            Self::Build(error) => write!(formatter, "could not build PiP compute pool: {error}"),
        }
    }
}

impl Error for ComputePoolError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ZeroThreads => None,
            Self::Build(error) => Some(error),
        }
    }
}

/// Explicit reusable Rayon pool for callers that want a fixed worker count.
///
/// Ordinary PiP operations do not require this type: they use the Rayon pool
/// in which they are called, or Rayon's global pool when no pool is installed.
pub struct ComputePool {
    inner: rayon::ThreadPool,
}

impl ComputePool {
    /// Constructs a reusable pool containing exactly `threads` workers.
    pub fn new(threads: usize) -> Result<Self, ComputePoolError> {
        if threads == 0 {
            return Err(ComputePoolError::ZeroThreads);
        }
        let inner = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(ComputePoolError::Build)?;
        Ok(Self { inner })
    }

    /// Returns the number of workers owned by this pool.
    pub fn threads(&self) -> usize {
        self.inner.current_num_threads()
    }

    /// Executes `operation` with this pool installed as the current Rayon pool.
    pub fn install<Operation, Output>(&self, operation: Operation) -> Output
    where
        Operation: FnOnce() -> Output + Send,
        Output: Send,
    {
        self.inner.install(operation)
    }
}

impl fmt::Debug for ComputePool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ComputePool")
            .field("threads", &self.threads())
            .finish_non_exhaustive()
    }
}

/// Creates a temporary fixed-size pool and executes one operation inside it.
///
/// Construct [`ComputePool`] directly when executing multiple operations so
/// that worker creation is amortized.
pub fn with_threads<Operation, Output>(
    threads: usize,
    operation: Operation,
) -> Result<Output, ComputePoolError>
where
    Operation: FnOnce() -> Output + Send,
    Output: Send,
{
    Ok(ComputePool::new(threads)?.install(operation))
}

/// Returns a chunk length that creates no more jobs than the current pool has
/// workers, additionally respecting an optional operation-local maximum.
#[inline]
pub(crate) fn parallel_chunk_len_with_max(
    work_units: usize,
    max_jobs: Option<usize>,
) -> Option<usize> {
    if work_units == 0 {
        return None;
    }
    let available = rayon::current_num_threads().max(1);
    let requested = max_jobs.unwrap_or(available).max(1);
    let jobs = available.min(requested).min(work_units);
    Some(work_units.div_ceil(jobs))
}

/// Returns a chunk length using all useful workers in the current pool.
#[inline]
pub(crate) fn parallel_chunk_len(work_units: usize) -> Option<usize> {
    parallel_chunk_len_with_max(work_units, None)
}

/// Reports whether adaptive elementwise work is large enough for Rayon.
#[inline]
pub(crate) fn should_parallelize_elements(elements: usize) -> bool {
    elements >= MIN_PARALLEL_ELEMENTS && rayon::current_num_threads() > 1
}

/// Reports whether a cost-aware kernel is large enough for Rayon.
#[inline]
pub(crate) fn should_parallelize_operations(operations: usize) -> bool {
    operations >= MIN_PARALLEL_OPERATIONS && rayon::current_num_threads() > 1
}

/// Minimum deterministic random lanes assigned to one Rayon job.
#[inline]
pub(crate) fn random_lanes_per_job(active_lanes: usize, max_threads: usize) -> usize {
    parallel_chunk_len_with_max(active_lanes, Some(max_threads)).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::{ComputePool, parallel_chunk_len_with_max, random_lanes_per_job};

    #[test]
    fn work_planning_respects_the_current_pool_and_operation_maximum() {
        let pool = ComputePool::new(6).unwrap();
        pool.install(|| {
            assert_eq!(parallel_chunk_len_with_max(100, None), Some(17));
            assert_eq!(parallel_chunk_len_with_max(100, Some(3)), Some(34));
            assert_eq!(parallel_chunk_len_with_max(100, Some(1)), Some(100));
            assert_eq!(random_lanes_per_job(32, 4), 8);
        });
    }
}

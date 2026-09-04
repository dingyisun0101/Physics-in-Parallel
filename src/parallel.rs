//! Process-wide parallel execution policy.
//!
//! PiP uses the current Rayon pool and never creates or owns one. Applications
//! should configure their Rayon pool to avoid oversubscription, then normally
//! call [`set_max_threads`] once near program startup. `None` leaves each PiP
//! method uncapped; `Some(n)` limits the jobs a single method can schedule.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const MIN_PARALLEL_ELEMENTS: usize = 16_384;
pub(crate) const MIN_PARALLEL_OPERATIONS: usize = 131_072;

// Zero is reserved for the public `None` setting.
static MAX_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Invalid process-wide parallel execution policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ParallelismError {
    ZeroThreads,
}

impl fmt::Display for ParallelismError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroThreads => formatter.write_str("maximum threads must be positive"),
        }
    }
}

impl std::error::Error for ParallelismError {}

/// Sets the maximum threads one PiP method may use.
///
/// `None` removes PiP's cap. `Some(0)` is invalid. This setting does not build
/// or resize a Rayon pool, so callers remain responsible for configuring the
/// pool used by their application and avoiding nested-pool oversubscription.
pub fn set_max_threads(max_threads: Option<usize>) -> Result<(), ParallelismError> {
    let encoded = match max_threads {
        None => 0,
        Some(0) => return Err(ParallelismError::ZeroThreads),
        Some(value) => value,
    };
    MAX_THREADS.store(encoded, Ordering::Relaxed);
    Ok(())
}

/// Returns the current process-wide PiP method cap.
pub fn max_threads() -> Option<usize> {
    match MAX_THREADS.load(Ordering::Relaxed) {
        0 => None,
        value => Some(value),
    }
}

/// Snapshot of the process policy and current Rayon pool for one operation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OperationBudget {
    jobs: usize,
}

impl OperationBudget {
    pub(crate) fn capture(work_units: usize) -> Option<Self> {
        if work_units == 0 {
            return None;
        }
        let available = rayon::current_num_threads().max(1);
        let requested = max_threads().unwrap_or(available);
        Some(Self {
            jobs: available.min(requested).min(work_units),
        })
    }

    pub(crate) fn chunk_len(self, work_units: usize) -> usize {
        work_units.div_ceil(self.jobs)
    }
}

/// Visits disjoint row-aligned chunks with one captured execution budget.
/// The callback receives the flat offset of each chunk. Small inputs use the
/// caller thread, while SIMD inside a callback remains independent of the cap.
pub(crate) fn for_each_chunk_mut<T: Send, F>(values: &mut [T], row_width: usize, function: F)
where
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    use rayon::prelude::*;
    assert!(row_width > 0 && values.len().is_multiple_of(row_width));
    let Some(budget) = OperationBudget::capture(values.len() / row_width) else {
        return;
    };
    if budget.jobs == 1 || values.len() < MIN_PARALLEL_ELEMENTS {
        function(0, values);
    } else {
        let chunk = budget.chunk_len(values.len() / row_width) * row_width;
        values
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(index, values)| {
                function(index * chunk, values);
            });
    }
}

#[inline]
pub(crate) fn parallel_chunk_len(work_units: usize) -> Option<usize> {
    OperationBudget::capture(work_units).map(|budget| budget.chunk_len(work_units))
}

#[inline]
pub(crate) fn should_parallelize_elements(elements: usize) -> bool {
    elements >= MIN_PARALLEL_ELEMENTS
        && OperationBudget::capture(elements).is_some_and(|budget| budget.jobs > 1)
}

#[inline]
pub(crate) fn should_parallelize_operations(operations: usize) -> bool {
    operations >= MIN_PARALLEL_OPERATIONS
        && OperationBudget::capture(operations).is_some_and(|budget| budget.jobs > 1)
}

#[inline]
pub(crate) fn random_lanes_per_job(active_lanes: usize) -> usize {
    parallel_chunk_len(active_lanes).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_and_round_trips_the_setting() {
        assert_eq!(set_max_threads(Some(0)), Err(ParallelismError::ZeroThreads));
        set_max_threads(Some(3)).unwrap();
        assert_eq!(max_threads(), Some(3));
        set_max_threads(None).unwrap();
        assert_eq!(max_threads(), None);
    }
}

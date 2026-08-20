//! Process-wide execution partition policy for PiP's internal Rayon work.

use std::error::Error;
use std::fmt;
use std::sync::OnceLock;

/// Default maximum number of PiP jobs created by one low-level operation.
pub const DEFAULT_PARALLEL_PARTITIONS: usize = 8;

static PARALLEL_PARTITIONS: OnceLock<usize> = OnceLock::new();

/// Failure while freezing PiP's process-wide execution partition count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ParallelismError {
    ZeroPartitions,
    AlreadyInitialized { configured: usize },
}

impl fmt::Display for ParallelismError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPartitions => formatter.write_str("PiP parallel partitions must be positive"),
            Self::AlreadyInitialized { configured } => write!(
                formatter,
                "PiP parallel partitions are already fixed at {configured}"
            ),
        }
    }
}

impl Error for ParallelismError {}

/// Sets PiP's process-wide low-level partition limit before its first use.
pub fn set_parallel_partitions(partitions: usize) -> Result<(), ParallelismError> {
    if partitions == 0 {
        return Err(ParallelismError::ZeroPartitions);
    }
    PARALLEL_PARTITIONS
        .set(partitions)
        .map_err(|_| ParallelismError::AlreadyInitialized {
            configured: parallel_partitions(),
        })
}

/// Returns the configured limit, freezing the default of eight on first use.
pub fn parallel_partitions() -> usize {
    *PARALLEL_PARTITIONS.get_or_init(|| DEFAULT_PARALLEL_PARTITIONS)
}

/// Returns a chunk length that creates at most the configured number of jobs.
#[inline]
pub(crate) fn parallel_chunk_len(work_units: usize) -> Option<usize> {
    (work_units > 0).then(|| work_units.div_ceil(parallel_partitions().min(work_units)))
}

/// Minimum number of deterministic lanes assigned to one Rayon job.
#[inline]
pub(crate) fn lanes_per_job(active_lanes: usize) -> usize {
    active_lanes.div_ceil(parallel_partitions().min(active_lanes))
}

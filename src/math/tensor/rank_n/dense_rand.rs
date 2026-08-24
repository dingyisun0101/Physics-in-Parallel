// src/math/tensor/rank_n/dense_rand.rs
/*!
Random filling for dense rank-N tensors.

`TensorRandFiller` owns a pool of RNG instances and splits tensor storage into
contiguous chunks during refresh. Seeded fillers are deterministic across
identical construction, RNG kind, and refresh sequences.

[`RngConfig`] controls the seed and method through the same interface used by
every other PiP stochastic component. Fill partitioning is a static backend
choice rather than scientific configuration.

Supported element/distribution pairs:
    - `f64`: `Uniform`, `Normal`, `Bernoulli`;
    - `i64`: `UniformInt`, `Bernoulli`;
    - `usize`: `UniformInt`;
    - `isize`: `UniformInt`.

Indexed fillers additionally expose exact uniform `usize` index sampling over
`0..upper` for algorithms that select sites or storage positions directly.

`new` and `try_new` construct stateful fillers. `try_new_indexed` constructs an
explicit-step filler. `RngConfig` selects deterministic seeding and the RNG
family; omitted values use documented defaults. [`NUM_RNGS`] fixes deterministic
random lanes. Each filler independently limits how many Rayon workers one fill
may occupy through [`TensorRandFiller::max_threads`].
*/

use std::{error::Error, fmt};

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_chacha::{ChaCha8Rng, ChaCha12Rng, ChaCha20Rng};
use rand_distr::{Bernoulli, Distribution, Normal, Uniform};
use rand_pcg::{Pcg64, Pcg64Mcg};
use rayon::prelude::*;

use crate::math::scalar::Scalar;
use crate::math::tensor::rank_n::dense::Tensor as DenseTensor;
use crate::math::tensor::rank_n::generic::{Dense, Tensor};
use crate::rng::{IndexedRng, RngConfig, RngConfigError, RngMethod};
use crate::threading::{parallel_chunk_len_with_max, random_lanes_per_job};

//===================================================================
// ---------------------------- Config ------------------------------
//===================================================================

/// Fixed deterministic RNG lane count used by stateful tensor fillers.
///
/// Execution parallelism is controlled independently, so changing a filler's
/// worker limit does not change the stateful random mapping.
pub const NUM_RNGS: usize = 32;

/// Default maximum workers that one tensor random filler may occupy.
pub const DEFAULT_RANDOM_MAX_THREADS: usize = 8;

//===================================================================
// -------------------------- Basic Types ---------------------------
//===================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RandType {
    Uniform { low: f64, high: f64 },    // floats: [low, high)
    UniformInt { low: i64, high: i64 }, // ints:   [low, high]
    Normal { mean: f64, std: f64 },
    Bernoulli { p: f64 },
}

impl RandType {
    #[inline]
    fn name(self) -> &'static str {
        match self {
            Self::Uniform { .. } => "Uniform",
            Self::UniformInt { .. } => "UniformInt",
            Self::Normal { .. } => "Normal",
            Self::Bernoulli { .. } => "Bernoulli",
        }
    }
}

/// Invalid tensor random-filler configuration or distribution.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorRandError {
    RngConfig(RngConfigError),
    UnsupportedDistribution {
        scalar: &'static str,
        distribution: &'static str,
    },
    InvalidUniformBounds {
        low: f64,
        high: f64,
    },
    InvalidNormalStd {
        std: f64,
    },
    InvalidBernoulliProbability {
        p: f64,
    },
    InvalidUniformIntBounds {
        low: i64,
        high: i64,
    },
    InvalidIndexUpperBound,
    InvalidMaxThreads,
    IntegerBoundsOutOfRange {
        scalar: &'static str,
        low: i64,
        high: i64,
    },
    IndexedStepRequired,
    StatefulMethodDoesNotSupportIndexedFill,
}

impl fmt::Display for TensorRandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RngConfig(error) => error.fmt(formatter),
            Self::UnsupportedDistribution {
                scalar,
                distribution,
            } => write!(
                formatter,
                "distribution {distribution} is not supported for scalar {scalar}"
            ),
            Self::InvalidUniformBounds { low, high } => write!(
                formatter,
                "uniform bounds must be finite with low < high; got low={low}, high={high}"
            ),
            Self::InvalidNormalStd { std } => {
                write!(
                    formatter,
                    "normal standard deviation must be finite and nonnegative; got {std}"
                )
            }
            Self::InvalidBernoulliProbability { p } => write!(
                formatter,
                "Bernoulli probability must be finite and in [0, 1]; got {p}"
            ),
            Self::InvalidUniformIntBounds { low, high } => write!(
                formatter,
                "integer uniform bounds require low <= high; got low={low}, high={high}"
            ),
            Self::InvalidIndexUpperBound => {
                write!(formatter, "uniform index upper bound must be positive")
            }
            Self::InvalidMaxThreads => {
                write!(
                    formatter,
                    "tensor random-filler max_threads must be positive"
                )
            }
            Self::IntegerBoundsOutOfRange { scalar, low, high } => write!(
                formatter,
                "integer uniform bounds [{low}, {high}] are outside scalar {scalar}"
            ),
            Self::IndexedStepRequired => write!(
                formatter,
                "indexed tensor random fillers require an explicit step and domain"
            ),
            Self::StatefulMethodDoesNotSupportIndexedFill => write!(
                formatter,
                "stateful tensor random fillers do not support explicit indexed fills"
            ),
        }
    }
}

impl Error for TensorRandError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RngConfig(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorRandFiller {
    kind: RandType,
    rng: RngConfig,
    num_rngs: usize,
    max_threads: usize,
    rngs: Vec<TensorRng>,
    indexed: Option<IndexedRng>,
}

impl TensorRandFiller {
    /// Constructs a filler from one unified random configuration.
    ///
    /// Missing values use host entropy, [`RngMethod::SmallRng`], and
    /// [`NUM_RNGS`], and [`DEFAULT_RANDOM_MAX_THREADS`]. Indexed methods are
    /// rejected.
    #[inline]
    pub fn new(kind: RandType, rng: RngConfig) -> Self {
        Self::try_new(kind, rng).expect("invalid tensor random filler configuration")
    }

    /// Fallibly constructs a filler from the same unified configuration as
    /// [`Self::new`].
    #[inline]
    pub fn try_new(kind: RandType, rng: RngConfig) -> Result<Self, TensorRandError> {
        let rng = rng
            .resolve_for(
                "TensorRandFiller",
                RngMethod::SmallRng,
                &[
                    RngMethod::Pcg64,
                    RngMethod::Pcg64Mcg,
                    RngMethod::SmallRng,
                    RngMethod::ChaCha8,
                    RngMethod::ChaCha12,
                    RngMethod::ChaCha20,
                ],
            )
            .map_err(TensorRandError::RngConfig)?;
        let req = NUM_RNGS;
        let rng_method = rng.method().expect("resolved RNG method");
        let seed = rng.seed().expect("resolved RNG seed");
        let mut master = SmallRng::seed_from_u64(seed);
        Ok(Self::from_master_rng(
            kind,
            rng,
            rng_method,
            req,
            &mut master,
        ))
    }

    /// Constructs a schedule-independent filler for explicit scientific steps.
    ///
    /// PiP keeps [`NUM_RNGS`] deterministic lanes while independently capping
    /// the number of Rayon jobs through this filler's `max_threads` value.
    pub fn try_new_indexed(kind: RandType, rng: RngConfig) -> Result<Self, TensorRandError> {
        let rng = rng
            .resolve_for(
                "TensorRandFiller",
                RngMethod::IndexedSplitMix64,
                &[RngMethod::IndexedSplitMix64],
            )
            .map_err(TensorRandError::RngConfig)?;
        Ok(Self {
            kind,
            rng,
            num_rngs: NUM_RNGS,
            max_threads: DEFAULT_RANDOM_MAX_THREADS,
            rngs: Vec::new(),
            indexed: Some(IndexedRng::from_resolved(rng)),
        })
    }

    fn from_master_rng(
        kind: RandType,
        rng: RngConfig,
        rng_method: RngMethod,
        num_rngs: usize,
        master: &mut SmallRng,
    ) -> Self {
        let mut rngs: Vec<TensorRng> = (0..num_rngs)
            .map(|_| TensorRng::from_master(rng_method, master))
            .collect();
        rngs.shrink_to_fit();
        Self {
            kind,
            rng,
            num_rngs,
            max_threads: DEFAULT_RANDOM_MAX_THREADS,
            rngs,
            indexed: None,
        }
    }

    #[inline]
    fn active_slices(&self, n: usize) -> usize {
        if n == 0 { 0 } else { self.num_rngs.min(n) }
    }

    #[inline]
    fn chunk_len(&self, n: usize, slices: usize) -> usize {
        if n == 0 || slices == 0 {
            0
        } else {
            n.div_ceil(slices)
        }
    }

    #[inline]
    fn chunk_plan(&self, n: usize) -> Option<(usize, usize)> {
        let slices = self.active_slices(n);
        if slices == 0 {
            None
        } else {
            Some((slices, self.chunk_len(n, slices)))
        }
    }

    #[inline]
    fn indexed_chunk_len(&self, n: usize) -> Option<usize> {
        parallel_chunk_len_with_max(n, Some(self.max_threads))
    }

    /// Refresh tensor values in-place.
    ///
    /// Panics:
    /// - when the filler distribution is invalid for `T`;
    /// - when distribution parameters are invalid.
    #[inline]
    pub fn refresh<T: TensorRandElement>(&mut self, tensor: &mut Tensor<T, Dense>) {
        self.try_refresh(tensor)
            .expect("invalid tensor random refresh configuration");
    }

    /// Fallibly refresh tensor values in-place.
    #[inline]
    pub fn try_refresh<T: TensorRandElement>(
        &mut self,
        tensor: &mut Tensor<T, Dense>,
    ) -> Result<(), TensorRandError> {
        if self.indexed.is_some() {
            return Err(TensorRandError::IndexedStepRequired);
        }
        T::try_fill(self, tensor.storage_mut())
    }

    pub(crate) fn try_refresh_dense<T: TensorRandElement>(
        &mut self,
        tensor: &mut DenseTensor<T>,
    ) -> Result<(), TensorRandError> {
        if self.indexed.is_some() {
            return Err(TensorRandError::IndexedStepRequired);
        }
        T::try_fill(self, tensor)
    }

    /// Fallibly fills a caller-owned contiguous slice in place.
    ///
    /// This uses the same distribution, RNG state, and parallel chunking as
    /// [`Self::try_refresh`] without requiring a tensor wrapper.
    #[inline]
    pub fn try_fill_slice<T: TensorRandElement>(
        &mut self,
        values: &mut [T],
    ) -> Result<(), TensorRandError> {
        if self.indexed.is_some() {
            return Err(TensorRandError::IndexedStepRequired);
        }
        T::try_fill_slice(self, values)
    }

    /// Fills a slice reproducibly for an explicit step and random domain.
    ///
    /// Results do not depend on Rayon scheduling or this filler's worker limit.
    pub fn try_fill_slice_at<T: TensorRandElement>(
        &self,
        values: &mut [T],
        step: u64,
        domain: u64,
    ) -> Result<(), TensorRandError> {
        let Some(key) = self.indexed else {
            return Err(TensorRandError::StatefulMethodDoesNotSupportIndexedFill);
        };
        T::try_fill_slice_indexed(self, key, values, step, domain, 1)
    }

    /// Fills row-major values using `(row, component)` random coordinates.
    pub fn try_fill_slice_at_layout<T: TensorRandElement>(
        &self,
        values: &mut [T],
        components: usize,
        step: u64,
        domain: u64,
    ) -> Result<(), TensorRandError> {
        assert!(components > 0, "indexed fill components must be positive");
        let Some(key) = self.indexed else {
            return Err(TensorRandError::StatefulMethodDoesNotSupportIndexedFill);
        };
        T::try_fill_slice_indexed(self, key, values, step, domain, components)
    }

    /// Fills exact uniform indices from `0..upper` for an explicit step.
    ///
    /// Sampling uses indexed Lemire rejection and is independent of Rayon
    /// scheduling. This operation is available only on indexed fillers and
    /// does not depend on the filler's scalar distribution.
    pub fn try_fill_indices_at(
        &self,
        values: &mut [usize],
        upper: usize,
        step: u64,
        domain: u64,
    ) -> Result<(), TensorRandError> {
        if upper == 0 {
            return Err(TensorRandError::InvalidIndexUpperBound);
        }
        let Some(key) = self.indexed else {
            return Err(TensorRandError::StatefulMethodDoesNotSupportIndexedFill);
        };
        let Some(chunk_len) = self.indexed_chunk_len(values.len()) else {
            return Ok(());
        };
        values
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let start = chunk_index * chunk_len;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    *value = key
                        .uniform_index(step, domain, (start + offset) as u64, 0, upper)
                        .expect("validated positive uniform-index upper bound");
                }
            });
        Ok(())
    }

    /// Samples two independent uniform indices per row and maps them directly
    /// into three caller-owned row-major buffers.
    ///
    /// This crate-internal bridge keeps indexed sampling inside the random
    /// backend while allowing consumers to assemble final records without
    /// allocating temporary index arrays.
    pub(crate) fn try_fill_index_pairs_with<T, F>(
        &self,
        buffers: (&mut [T], &mut [T], &mut [T]),
        components: usize,
        upper: usize,
        step: u64,
        domains: (u64, u64),
        map: F,
    ) -> Result<(), TensorRandError>
    where
        T: Send,
        F: Fn(usize, usize, &mut [T], &mut [T], &mut [T]) + Send + Sync,
    {
        let (first, second, third) = buffers;
        let (first_domain, second_domain) = domains;
        assert!(components > 0, "indexed pair components must be positive");
        assert_eq!(first.len(), second.len(), "indexed pair buffer mismatch");
        assert_eq!(first.len(), third.len(), "indexed pair buffer mismatch");
        assert_eq!(
            first.len() % components,
            0,
            "indexed pair buffers must contain complete rows"
        );
        if upper == 0 {
            return Err(TensorRandError::InvalidIndexUpperBound);
        }
        let Some(key) = self.indexed else {
            return Err(TensorRandError::StatefulMethodDoesNotSupportIndexedFill);
        };
        let rows = first.len() / components;
        let min_rows_per_job = self.indexed_chunk_len(rows).unwrap_or(1);
        first
            .par_chunks_mut(components)
            .zip(second.par_chunks_mut(components))
            .zip(third.par_chunks_mut(components))
            .with_min_len(min_rows_per_job)
            .enumerate()
            .for_each(|(row, ((first, second), third))| {
                let first_index = key
                    .uniform_index(step, first_domain, row as u64, 0, upper)
                    .expect("validated positive uniform-index upper bound");
                let second_index = key
                    .uniform_index(step, second_domain, row as u64, 0, upper)
                    .expect("validated positive uniform-index upper bound");
                map(first_index, second_index, first, second, third);
            });
        Ok(())
    }

    #[inline]
    pub fn kind(&self) -> &RandType {
        &self.kind
    }

    #[inline]
    pub fn set_kind(&mut self, kind: RandType) {
        self.kind = kind;
    }

    #[inline]
    pub fn rng_config(&self) -> RngConfig {
        self.rng
    }

    /// Returns the maximum number of workers that one fill may occupy.
    ///
    /// The current Rayon pool may provide fewer workers. This runtime limit
    /// does not change deterministic RNG lanes or generated values.
    #[inline]
    pub const fn max_threads(&self) -> usize {
        self.max_threads
    }

    /// Sets the maximum number of workers that one fill may occupy.
    pub fn set_max_threads(&mut self, max_threads: usize) -> Result<(), TensorRandError> {
        if max_threads == 0 {
            return Err(TensorRandError::InvalidMaxThreads);
        }
        self.max_threads = max_threads;
        Ok(())
    }

    /// Returns this filler with a new per-fill worker maximum.
    pub fn with_max_threads(mut self, max_threads: usize) -> Result<Self, TensorRandError> {
        self.set_max_threads(max_threads)?;
        Ok(self)
    }
}

fn unsupported<T: 'static>(kind: RandType) -> TensorRandError {
    TensorRandError::UnsupportedDistribution {
        scalar: core::any::type_name::<T>(),
        distribution: kind.name(),
    }
}

#[derive(Debug, Clone)]
enum TensorRng {
    SmallRng(SmallRng),
    Pcg64Mcg(Pcg64Mcg),
    Pcg64(Pcg64),
    ChaCha8(ChaCha8Rng),
    ChaCha12(ChaCha12Rng),
    ChaCha20(ChaCha20Rng),
}

impl TensorRng {
    fn from_master(kind: RngMethod, master: &mut SmallRng) -> Self {
        match kind {
            RngMethod::SmallRng => Self::SmallRng(SmallRng::from_rng(master)),
            RngMethod::Pcg64Mcg => Self::Pcg64Mcg(Pcg64Mcg::from_rng(master)),
            RngMethod::Pcg64 => Self::Pcg64(Pcg64::from_rng(master)),
            RngMethod::ChaCha8 => Self::ChaCha8(ChaCha8Rng::from_rng(master)),
            RngMethod::ChaCha12 => Self::ChaCha12(ChaCha12Rng::from_rng(master)),
            RngMethod::ChaCha20 => Self::ChaCha20(ChaCha20Rng::from_rng(master)),
            RngMethod::IndexedSplitMix64 => unreachable!("indexed RNG rejected during resolve"),
        }
    }

    fn fill_sample<T, D>(&mut self, chunk: &mut [T], dist: &D)
    where
        D: Distribution<T>,
    {
        match self {
            Self::SmallRng(rng) => fill_sample_with_rng(chunk, dist, rng),
            Self::Pcg64Mcg(rng) => fill_sample_with_rng(chunk, dist, rng),
            Self::Pcg64(rng) => fill_sample_with_rng(chunk, dist, rng),
            Self::ChaCha8(rng) => fill_sample_with_rng(chunk, dist, rng),
            Self::ChaCha12(rng) => fill_sample_with_rng(chunk, dist, rng),
            Self::ChaCha20(rng) => fill_sample_with_rng(chunk, dist, rng),
        }
    }

    fn fill_mapped_sample<T, S, D, F>(&mut self, chunk: &mut [T], dist: &D, map: F)
    where
        D: Distribution<S>,
        F: Fn(S) -> T + Copy,
    {
        match self {
            Self::SmallRng(rng) => fill_mapped_sample_with_rng(chunk, dist, map, rng),
            Self::Pcg64Mcg(rng) => fill_mapped_sample_with_rng(chunk, dist, map, rng),
            Self::Pcg64(rng) => fill_mapped_sample_with_rng(chunk, dist, map, rng),
            Self::ChaCha8(rng) => fill_mapped_sample_with_rng(chunk, dist, map, rng),
            Self::ChaCha12(rng) => fill_mapped_sample_with_rng(chunk, dist, map, rng),
            Self::ChaCha20(rng) => fill_mapped_sample_with_rng(chunk, dist, map, rng),
        }
    }
}

fn fill_sample_with_rng<T, D, R>(chunk: &mut [T], dist: &D, rng: &mut R)
where
    D: Distribution<T>,
    R: rand::Rng + ?Sized,
{
    for x in chunk {
        *x = dist.sample(rng);
    }
}

fn fill_mapped_sample_with_rng<T, S, D, F, R>(chunk: &mut [T], dist: &D, map: F, rng: &mut R)
where
    D: Distribution<S>,
    F: Fn(S) -> T + Copy,
    R: rand::Rng + ?Sized,
{
    for x in chunk {
        *x = map(dist.sample(rng));
    }
}

//===================================================================
// ------------- Sealed trait for per-type specialization -----------
//===================================================================

mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for i64 {}
    impl Sealed for usize {}
    impl Sealed for isize {}
}

pub trait TensorRandElement: sealed::Sealed + Sized + Scalar {
    fn try_fill_slice(
        filler: &mut TensorRandFiller,
        values: &mut [Self],
    ) -> Result<(), TensorRandError>;

    fn try_fill_slice_indexed(
        filler: &TensorRandFiller,
        key: IndexedRng,
        values: &mut [Self],
        step: u64,
        domain: u64,
        components: usize,
    ) -> Result<(), TensorRandError>;

    fn try_fill(
        filler: &mut TensorRandFiller,
        tensor: &mut DenseTensor<Self>,
    ) -> Result<(), TensorRandError> {
        Self::try_fill_slice(filler, tensor.data_mut())
    }

    #[inline]
    fn fill(filler: &mut TensorRandFiller, tensor: &mut DenseTensor<Self>) {
        Self::try_fill(filler, tensor).expect("invalid tensor random refresh configuration");
    }
}

// ---------------------------- f64 ---------------------------------
impl TensorRandElement for f64 {
    fn try_fill_slice_indexed(
        filler: &TensorRandFiller,
        key: IndexedRng,
        values: &mut [f64],
        step: u64,
        domain: u64,
        components: usize,
    ) -> Result<(), TensorRandError> {
        let Some(chunk_len) = filler.indexed_chunk_len(values.len()) else {
            return Ok(());
        };
        match filler.kind {
            RandType::Uniform { low, high } => {
                if !(low.is_finite() && high.is_finite() && low < high) {
                    return Err(TensorRandError::InvalidUniformBounds { low, high });
                }
                values
                    .par_chunks_mut(chunk_len)
                    .enumerate()
                    .for_each(|(chunk_index, chunk)| {
                        let start = chunk_index * chunk_len;
                        for (offset, value) in chunk.iter_mut().enumerate() {
                            let index = start + offset;
                            let unit = key.unit_f64(
                                step,
                                domain,
                                (index / components) as u64,
                                (index % components) as u64,
                                0,
                            );
                            *value = low + (high - low) * unit;
                        }
                    });
            }
            RandType::Normal { mean, std } => {
                if !(std.is_finite() && std >= 0.0) {
                    return Err(TensorRandError::InvalidNormalStd { std });
                }
                if std == 0.0 {
                    values
                        .par_chunks_mut(chunk_len)
                        .for_each(|chunk| chunk.fill(mean));
                    return Ok(());
                }
                values
                    .par_chunks_mut(chunk_len)
                    .enumerate()
                    .for_each(|(chunk_index, chunk)| {
                        let start = chunk_index * chunk_len;
                        for (offset, value) in chunk.iter_mut().enumerate() {
                            let index = start + offset;
                            *value = mean
                                + std
                                    * key.standard_normal(
                                        step,
                                        domain,
                                        (index / components) as u64,
                                        (index % components) as u64,
                                    );
                        }
                    });
            }
            RandType::Bernoulli { p } => {
                if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
                    return Err(TensorRandError::InvalidBernoulliProbability { p });
                }
                values
                    .par_chunks_mut(chunk_len)
                    .enumerate()
                    .for_each(|(chunk_index, chunk)| {
                        let start = chunk_index * chunk_len;
                        for (offset, value) in chunk.iter_mut().enumerate() {
                            let index = start + offset;
                            *value = if key.unit_f64(
                                step,
                                domain,
                                (index / components) as u64,
                                (index % components) as u64,
                                0,
                            ) < p
                            {
                                1.0
                            } else {
                                0.0
                            };
                        }
                    });
            }
            kind => return Err(unsupported::<f64>(kind)),
        }
        Ok(())
    }

    fn try_fill_slice(
        filler: &mut TensorRandFiller,
        values: &mut [f64],
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(values.len()) else {
            return Ok(());
        };
        let rngs = &mut filler.rngs[..slices];
        let lanes_per_job = random_lanes_per_job(slices, filler.max_threads);

        match filler.kind {
            RandType::Uniform { low, high } => {
                let dist = Uniform::new(low, high)
                    .map_err(|_| TensorRandError::InvalidUniformBounds { low, high })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .with_min_len(lanes_per_job)
                    .for_each(|(chunk, rng)| rng.fill_sample(chunk, &dist));
            }
            RandType::Normal { mean, std } => {
                if !(std.is_finite() && std >= 0.0) {
                    return Err(TensorRandError::InvalidNormalStd { std });
                }
                if std == 0.0 {
                    values
                        .par_chunks_mut(chunk_len)
                        .with_min_len(lanes_per_job)
                        .for_each(|chunk| chunk.fill(mean));
                    return Ok(());
                }
                let dist = Normal::new(mean, std)
                    .map_err(|_| TensorRandError::InvalidNormalStd { std })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .with_min_len(lanes_per_job)
                    .for_each(|(chunk, rng)| rng.fill_sample(chunk, &dist));
            }
            RandType::Bernoulli { p } => {
                let dist = Bernoulli::new(p)
                    .map_err(|_| TensorRandError::InvalidBernoulliProbability { p })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .with_min_len(lanes_per_job)
                    .for_each(|(chunk, rng)| {
                        rng.fill_mapped_sample(chunk, &dist, |x| if x { 1.0 } else { 0.0 })
                    });
            }
            kind => return Err(unsupported::<f64>(kind)),
        }

        Ok(())
    }
}

// ---------------------------- i64 ---------------------------------
impl TensorRandElement for i64 {
    fn try_fill_slice_indexed(
        _filler: &TensorRandFiller,
        _key: IndexedRng,
        _values: &mut [i64],
        _step: u64,
        _domain: u64,
        _components: usize,
    ) -> Result<(), TensorRandError> {
        Err(TensorRandError::UnsupportedDistribution {
            scalar: "i64",
            distribution: "indexed",
        })
    }

    fn try_fill_slice(
        filler: &mut TensorRandFiller,
        values: &mut [i64],
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(values.len()) else {
            return Ok(());
        };
        let rngs = &mut filler.rngs[..slices];
        let lanes_per_job = random_lanes_per_job(slices, filler.max_threads);

        match filler.kind {
            RandType::UniformInt { low, high } => {
                let dist = Uniform::new_inclusive(low, high)
                    .map_err(|_| TensorRandError::InvalidUniformIntBounds { low, high })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .with_min_len(lanes_per_job)
                    .for_each(|(chunk, rng)| rng.fill_sample(chunk, &dist));
            }
            RandType::Bernoulli { p } => {
                let dist = Bernoulli::new(p)
                    .map_err(|_| TensorRandError::InvalidBernoulliProbability { p })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .with_min_len(lanes_per_job)
                    .for_each(|(chunk, rng)| {
                        rng.fill_mapped_sample(chunk, &dist, |x| if x { 1 } else { 0 })
                    });
            }
            kind => return Err(unsupported::<i64>(kind)),
        }

        Ok(())
    }
}

// ---------------------------- usize -------------------------------
impl TensorRandElement for usize {
    fn try_fill_slice_indexed(
        _filler: &TensorRandFiller,
        _key: IndexedRng,
        _values: &mut [usize],
        _step: u64,
        _domain: u64,
        _components: usize,
    ) -> Result<(), TensorRandError> {
        Err(TensorRandError::UnsupportedDistribution {
            scalar: "usize",
            distribution: "indexed",
        })
    }

    fn try_fill_slice(
        filler: &mut TensorRandFiller,
        values: &mut [usize],
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(values.len()) else {
            return Ok(());
        };
        let rngs = &mut filler.rngs[..slices];
        let lanes_per_job = random_lanes_per_job(slices, filler.max_threads);

        match filler.kind {
            RandType::UniformInt { low, high } => {
                let (low_u, high_u) = match (usize::try_from(low), usize::try_from(high)) {
                    (Ok(lo), Ok(hi)) if lo <= hi => (lo, hi),
                    _ => {
                        return Err(TensorRandError::IntegerBoundsOutOfRange {
                            scalar: "usize",
                            low,
                            high,
                        });
                    }
                };
                let dist = Uniform::new_inclusive(low_u, high_u)
                    .map_err(|_| TensorRandError::InvalidUniformIntBounds { low, high })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .with_min_len(lanes_per_job)
                    .for_each(|(chunk, rng)| rng.fill_sample(chunk, &dist));
            }
            kind => return Err(unsupported::<usize>(kind)),
        }

        Ok(())
    }
}

// ---------------------------- isize -------------------------------
impl TensorRandElement for isize {
    fn try_fill_slice_indexed(
        _filler: &TensorRandFiller,
        _key: IndexedRng,
        _values: &mut [isize],
        _step: u64,
        _domain: u64,
        _components: usize,
    ) -> Result<(), TensorRandError> {
        Err(TensorRandError::UnsupportedDistribution {
            scalar: "isize",
            distribution: "indexed",
        })
    }

    fn try_fill_slice(
        filler: &mut TensorRandFiller,
        values: &mut [isize],
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(values.len()) else {
            return Ok(());
        };
        let rngs = &mut filler.rngs[..slices];
        let lanes_per_job = random_lanes_per_job(slices, filler.max_threads);

        match filler.kind {
            RandType::UniformInt { low, high } => {
                if isize::try_from(low).is_err() || isize::try_from(high).is_err() {
                    return Err(TensorRandError::IntegerBoundsOutOfRange {
                        scalar: "isize",
                        low,
                        high,
                    });
                }
                let dist = Uniform::<i64>::new_inclusive(low, high)
                    .map_err(|_| TensorRandError::InvalidUniformIntBounds { low, high })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .with_min_len(lanes_per_job)
                    .for_each(|(chunk, rng)| rng.fill_mapped_sample(chunk, &dist, |x| x as isize));
            }
            kind => return Err(unsupported::<isize>(kind)),
        }

        Ok(())
    }
}

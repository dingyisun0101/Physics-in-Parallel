// src/math/tensor/rank_n/dense_rand.rs
/*!
Random filling for dense rank-N tensors.

`TensorRandFiller` owns a pool of RNG instances and splits tensor storage into
contiguous chunks during refresh. Seeded fillers are deterministic across
identical construction, RNG kind, and refresh sequences.

[`RngConfig`] controls the seed, method, and parallel stream count through the
same interface used by every other PiP stochastic component.

Supported element/distribution pairs:
    - `f64`: `Uniform`, `Normal`, `Bernoulli`;
    - `i64`: `UniformInt`, `Bernoulli`;
    - `usize`: `UniformInt`;
    - `isize`: `UniformInt`.

`new` and `try_new` are the only construction interfaces. Optional arguments
select deterministic seeding, the RNG family, and stream count; omitted values
use the documented defaults.
*/

use std::num::NonZeroUsize;
use std::{error::Error, fmt};

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_chacha::{ChaCha8Rng, ChaCha12Rng, ChaCha20Rng};
use rand_distr::{Bernoulli, Distribution, Normal, Uniform};
use rand_pcg::{Pcg64, Pcg64Mcg};
use rayon::prelude::*;

use crate::math::scalar::Scalar;
use crate::math::tensor::dense::Tensor;
use crate::rng::{RngConfig, RngConfigError, RngMethod};

//===================================================================
// ---------------------------- Config ------------------------------
//===================================================================

pub const NUM_RNGS: usize = 64;

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
    IntegerBoundsOutOfRange {
        scalar: &'static str,
        low: i64,
        high: i64,
    },
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
                    "normal standard deviation must be finite and positive; got {std}"
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
            Self::IntegerBoundsOutOfRange { scalar, low, high } => write!(
                formatter,
                "integer uniform bounds [{low}, {high}] are outside scalar {scalar}"
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
    rngs: Vec<TensorRng>,
}

impl TensorRandFiller {
    /// Constructs a filler from one unified random configuration.
    ///
    /// Missing values use host entropy, [`RngMethod::SmallRng`], and
    /// [`NUM_RNGS`]. Indexed methods are rejected.
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
                NonZeroUsize::new(NUM_RNGS),
            )
            .map_err(TensorRandError::RngConfig)?;
        let req = rng.parallel_streams().map_or(NUM_RNGS, |count| count.get());
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
            rngs,
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

    /// Refresh tensor values in-place.
    ///
    /// Panics:
    ///	- when the filler distribution is invalid for `T`;
    ///	- when distribution parameters are invalid.
    #[inline]
    pub fn refresh<T: TensorRandElement>(&mut self, tensor: &mut Tensor<T>) {
        self.try_refresh(tensor)
            .expect("invalid tensor random refresh configuration");
    }

    /// Fallibly refresh tensor values in-place.
    #[inline]
    pub fn try_refresh<T: TensorRandElement>(
        &mut self,
        tensor: &mut Tensor<T>,
    ) -> Result<(), TensorRandError> {
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
        T::try_fill_slice(self, values)
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

    fn try_fill(
        filler: &mut TensorRandFiller,
        tensor: &mut Tensor<Self>,
    ) -> Result<(), TensorRandError> {
        Self::try_fill_slice(filler, tensor.data_mut())
    }

    #[inline]
    fn fill(filler: &mut TensorRandFiller, tensor: &mut Tensor<Self>) {
        Self::try_fill(filler, tensor).expect("invalid tensor random refresh configuration");
    }
}

// ---------------------------- f64 ---------------------------------
impl TensorRandElement for f64 {
    fn try_fill_slice(
        filler: &mut TensorRandFiller,
        values: &mut [f64],
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(values.len()) else {
            return Ok(());
        };
        let rngs = &mut filler.rngs[..slices];

        match filler.kind {
            RandType::Uniform { low, high } => {
                let dist = Uniform::new(low, high)
                    .map_err(|_| TensorRandError::InvalidUniformBounds { low, high })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| rng.fill_sample(chunk, &dist));
            }
            RandType::Normal { mean, std } => {
                if !(std.is_finite() && std > 0.0) {
                    return Err(TensorRandError::InvalidNormalStd { std });
                }
                let dist = Normal::new(mean, std)
                    .map_err(|_| TensorRandError::InvalidNormalStd { std })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| rng.fill_sample(chunk, &dist));
            }
            RandType::Bernoulli { p } => {
                let dist = Bernoulli::new(p)
                    .map_err(|_| TensorRandError::InvalidBernoulliProbability { p })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
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
    fn try_fill_slice(
        filler: &mut TensorRandFiller,
        values: &mut [i64],
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(values.len()) else {
            return Ok(());
        };
        let rngs = &mut filler.rngs[..slices];

        match filler.kind {
            RandType::UniformInt { low, high } => {
                let dist = Uniform::new_inclusive(low, high)
                    .map_err(|_| TensorRandError::InvalidUniformIntBounds { low, high })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| rng.fill_sample(chunk, &dist));
            }
            RandType::Bernoulli { p } => {
                let dist = Bernoulli::new(p)
                    .map_err(|_| TensorRandError::InvalidBernoulliProbability { p })?;
                values
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
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
    fn try_fill_slice(
        filler: &mut TensorRandFiller,
        values: &mut [usize],
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(values.len()) else {
            return Ok(());
        };
        let rngs = &mut filler.rngs[..slices];

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
                    .for_each(|(chunk, rng)| rng.fill_sample(chunk, &dist));
            }
            kind => return Err(unsupported::<usize>(kind)),
        }

        Ok(())
    }
}

// ---------------------------- isize -------------------------------
impl TensorRandElement for isize {
    fn try_fill_slice(
        filler: &mut TensorRandFiller,
        values: &mut [isize],
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(values.len()) else {
            return Ok(());
        };
        let rngs = &mut filler.rngs[..slices];

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
                    .for_each(|(chunk, rng)| rng.fill_mapped_sample(chunk, &dist, |x| x as isize));
            }
            kind => return Err(unsupported::<isize>(kind)),
        }

        Ok(())
    }
}

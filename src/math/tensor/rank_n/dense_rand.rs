// src/math/tensor/rank_n/dense_rand.rs
/*!
Random filling for dense rank-N tensors.

`TensorRandFiller` owns a pool of RNG instances and splits tensor storage into
contiguous chunks during refresh. Seeded fillers are deterministic across
identical construction, RNG kind, and refresh sequences.

`RngKind` controls the concrete RNG family. When the creation API receives
`None`, it defaults to `SmallRng`.

Supported element/distribution pairs:
    - `f64`: `Uniform`, `Normal`, `Bernoulli`;
    - `i64`: `UniformInt`, `Bernoulli`;
    - `usize`: `UniformInt`;
    - `isize`: `UniformInt`.

`new` and `try_new` are the only construction interfaces. Optional arguments
select deterministic seeding, the RNG family, and stream count; omitted values
use the documented defaults.
*/

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_chacha::{ChaCha8Rng, ChaCha12Rng, ChaCha20Rng};
use rand_distr::{Bernoulli, Distribution, Normal, Uniform};
use rand_pcg::{Pcg64, Pcg64Mcg};
use rayon::prelude::*;

use crate::math::scalar::Scalar;
use crate::math::tensor::dense::Tensor;

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

/// RNG families available to `TensorRandFiller`.
///
/// Ordering:
///	- fastest to slowest by measured tensor-fill speed;
///	- benchmark basis: `f64` `Uniform(-1, 1)`, 120,000,000 elements,
///	  80 RNG chunks, release build, on the local Xeon Gold 6148 machine.
///
/// Default behavior:
///	- constructor `rng_kind == None` maps to `SmallRng`, which is the concise
///	  general-purpose default chosen for predictable user-facing behavior even
///	  though the enum itself is ordered by measured fill speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngKind {
    Pcg64,
    Pcg64Mcg,
    SmallRng,
    ChaCha8,
    ChaCha12,
    ChaCha20,
}

impl Default for RngKind {
    #[inline]
    fn default() -> Self {
        Self::SmallRng
    }
}

impl RngKind {
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::Pcg64 => "Pcg64",
            Self::Pcg64Mcg => "Pcg64Mcg",
            Self::SmallRng => "SmallRng",
            Self::ChaCha8 => "ChaCha8",
            Self::ChaCha12 => "ChaCha12",
            Self::ChaCha20 => "ChaCha20",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "pcg64" => Some(Self::Pcg64),
            "pcg64mcg" | "pcg64_mcg" | "pcg64-fast" | "pcg64fast" => Some(Self::Pcg64Mcg),
            "small" | "smallrng" => Some(Self::SmallRng),
            "chacha8" | "chacha8rng" => Some(Self::ChaCha8),
            "chacha12" | "chacha12rng" => Some(Self::ChaCha12),
            "chacha20" | "chacha20rng" | "chacha" => Some(Self::ChaCha20),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorRandError {
    ZeroRngCount,
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

#[derive(Debug, Clone)]
pub struct TensorRandFiller {
    kind: RandType,
    seed: u64,
    rng_kind: RngKind,
    num_rngs: usize,
    rngs: Vec<TensorRng>,
}

impl TensorRandFiller {
    /// Constructs a filler from one unified random configuration.
    ///
    /// `seed == None` obtains entropy from the host, `rng_kind == None` uses
    /// [`RngKind::default`], and `num_rngs == None` uses [`NUM_RNGS`].
    /// Panics when the supplied stream count is zero.
    #[inline]
    pub fn new(
        kind: RandType,
        seed: Option<u64>,
        rng_kind: Option<RngKind>,
        num_rngs: Option<usize>,
    ) -> Self {
        Self::try_new(kind, seed, rng_kind, num_rngs)
            .expect("invalid tensor random filler configuration")
    }

    /// Fallibly constructs a filler from the same unified configuration as
    /// [`Self::new`].
    #[inline]
    pub fn try_new(
        kind: RandType,
        seed: Option<u64>,
        rng_kind: Option<RngKind>,
        num_rngs: Option<usize>,
    ) -> Result<Self, TensorRandError> {
        let req = rng_count(num_rngs)?;
        let rng_kind = rng_kind.unwrap_or_default();
        let seed = seed.unwrap_or_else(rand::random);
        let mut master = SmallRng::seed_from_u64(seed);
        Ok(Self::from_master_rng(
            kind,
            seed,
            rng_kind,
            req,
            &mut master,
        ))
    }

    fn from_master_rng(
        kind: RandType,
        seed: u64,
        rng_kind: RngKind,
        num_rngs: usize,
        master: &mut SmallRng,
    ) -> Self {
        let mut rngs: Vec<TensorRng> = (0..num_rngs)
            .map(|_| TensorRng::from_master(rng_kind, master))
            .collect();
        rngs.shrink_to_fit();
        Self {
            kind,
            seed,
            rng_kind,
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

    #[inline]
    pub fn kind(&self) -> &RandType {
        &self.kind
    }

    #[inline]
    pub fn set_kind(&mut self, kind: RandType) {
        self.kind = kind;
    }

    #[inline]
    pub fn rng_kind(&self) -> RngKind {
        self.rng_kind
    }

    /// Returns the resolved root seed, including a host-generated seed when
    /// construction received `None`.
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

fn rng_count(num_rngs: Option<usize>) -> Result<usize, TensorRandError> {
    match num_rngs {
        Some(0) => Err(TensorRandError::ZeroRngCount),
        Some(n) => Ok(n),
        None => Ok(NUM_RNGS),
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
    fn from_master(kind: RngKind, master: &mut SmallRng) -> Self {
        match kind {
            RngKind::SmallRng => Self::SmallRng(SmallRng::from_rng(master)),
            RngKind::Pcg64Mcg => Self::Pcg64Mcg(Pcg64Mcg::from_rng(master)),
            RngKind::Pcg64 => Self::Pcg64(Pcg64::from_rng(master)),
            RngKind::ChaCha8 => Self::ChaCha8(ChaCha8Rng::from_rng(master)),
            RngKind::ChaCha12 => Self::ChaCha12(ChaCha12Rng::from_rng(master)),
            RngKind::ChaCha20 => Self::ChaCha20(ChaCha20Rng::from_rng(master)),
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
    fn try_fill(
        filler: &mut TensorRandFiller,
        tensor: &mut Tensor<Self>,
    ) -> Result<(), TensorRandError>;

    #[inline]
    fn fill(filler: &mut TensorRandFiller, tensor: &mut Tensor<Self>) {
        Self::try_fill(filler, tensor).expect("invalid tensor random refresh configuration");
    }
}

// ---------------------------- f64 ---------------------------------
impl TensorRandElement for f64 {
    fn try_fill(
        filler: &mut TensorRandFiller,
        tensor: &mut Tensor<f64>,
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(tensor.data().len()) else {
            return Ok(());
        };
        let rngs = &mut filler.rngs[..slices];

        match filler.kind {
            RandType::Uniform { low, high } => {
                let dist = Uniform::new(low, high)
                    .map_err(|_| TensorRandError::InvalidUniformBounds { low, high })?;
                tensor
                    .data_mut()
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
                tensor
                    .data_mut()
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| rng.fill_sample(chunk, &dist));
            }
            RandType::Bernoulli { p } => {
                let dist = Bernoulli::new(p)
                    .map_err(|_| TensorRandError::InvalidBernoulliProbability { p })?;
                tensor
                    .data_mut()
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
    fn try_fill(
        filler: &mut TensorRandFiller,
        tensor: &mut Tensor<i64>,
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(tensor.data().len()) else {
            return Ok(());
        };
        let rngs = &mut filler.rngs[..slices];

        match filler.kind {
            RandType::UniformInt { low, high } => {
                let dist = Uniform::new_inclusive(low, high)
                    .map_err(|_| TensorRandError::InvalidUniformIntBounds { low, high })?;
                tensor
                    .data_mut()
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| rng.fill_sample(chunk, &dist));
            }
            RandType::Bernoulli { p } => {
                let dist = Bernoulli::new(p)
                    .map_err(|_| TensorRandError::InvalidBernoulliProbability { p })?;
                tensor
                    .data_mut()
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
    fn try_fill(
        filler: &mut TensorRandFiller,
        tensor: &mut Tensor<usize>,
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(tensor.data().len()) else {
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
                tensor
                    .data_mut()
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
    fn try_fill(
        filler: &mut TensorRandFiller,
        tensor: &mut Tensor<isize>,
    ) -> Result<(), TensorRandError> {
        let Some((slices, chunk_len)) = filler.chunk_plan(tensor.data().len()) else {
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
                tensor
                    .data_mut()
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| rng.fill_mapped_sample(chunk, &dist, |x| x as isize));
            }
            kind => return Err(unsupported::<isize>(kind)),
        }

        Ok(())
    }
}

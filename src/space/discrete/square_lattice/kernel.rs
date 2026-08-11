/*!
Deterministic displacement kernels for square-lattice pair generation.

A kernel maps indexed random values to a move length or nearest-neighbor
direction code. Sampling is a pure function of an explicit key, sweep, and
sample index, so output does not depend on Rayon worker scheduling.
*/

use std::error::Error;
use std::fmt;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::random::{DOMAIN_KERNEL_SAMPLE, RandomKey, uniform_index, unit_f64};

/// Serializable description of one square-lattice displacement distribution.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Serialize)]
pub enum KernelType {
    PowerLaw { l: f64, c: f64, mu: f64 },
    Uniform { l: f64, c: f64 },
    NearestNeighbor { d: usize },
}

/// Invalid displacement-kernel configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum KernelError {
    InvalidPowerLaw { l: f64, c: f64, mu: f64 },
    InvalidUniform { l: f64, c: f64 },
    InvalidNearestNeighborDimension { dimension: usize },
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidPowerLaw { l, c, mu } => write!(
                formatter,
                "power-law kernel requires finite l > c > 0 and finite mu > 0; got l={l}, c={c}, mu={mu}"
            ),
            Self::InvalidUniform { l, c } => write!(
                formatter,
                "uniform kernel requires finite l > c; got l={l}, c={c}"
            ),
            Self::InvalidNearestNeighborDimension { dimension } => write!(
                formatter,
                "nearest-neighbor kernel dimension must be positive; got {dimension}"
            ),
        }
    }
}

impl Error for KernelError {}

/// Indexed displacement distribution used by [`RandPairGenerator`](super::RandPairGenerator).
pub trait Kernel: Send + Sync {
    /// Samples one value from the coordinate `(key, sweep, sample_index)`.
    fn sample_indexed(&self, key: RandomKey, sweep: u64, sample_index: u64) -> f64;

    /// Samples a stable batch whose result does not depend on worker count.
    fn sample_batch_indexed(&self, key: RandomKey, sweep: u64, n: usize) -> Vec<f64> {
        let mut output = vec![0.0; n];
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| {
                *value = self.sample_indexed(key, sweep, index as u64);
            });
        output
    }

    /// Returns the compact configuration for this distribution.
    fn kind(&self) -> KernelType;

    /// Clones this kernel behind a trait object.
    fn boxed_clone(&self) -> Box<dyn Kernel>;
}

impl Clone for Box<dyn Kernel> {
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

/// Creates a validated kernel without panicking on user configuration.
pub fn try_create_kernel(kernel_type: KernelType) -> Result<Box<dyn Kernel>, KernelError> {
    match kernel_type {
        KernelType::PowerLaw { l, c, mu } => Ok(Box::new(PowerLawKernel::try_new(l, c, mu)?)),
        KernelType::Uniform { l, c } => Ok(Box::new(UniformKernel::try_new(l, c)?)),
        KernelType::NearestNeighbor { d } => Ok(Box::new(NearestNeighborKernel::try_new(d)?)),
    }
}

/// Creates a kernel, panicking only as a compatibility convenience.
///
/// New callers should prefer [`try_create_kernel`].
pub fn create_kernel(kernel_type: KernelType) -> Box<dyn Kernel> {
    try_create_kernel(kernel_type).expect("invalid square-lattice kernel configuration")
}

#[derive(Debug, Clone)]
pub struct PowerLawKernel {
    kind: KernelType,
    l_pow: f64,
    c_pow: f64,
}

impl PowerLawKernel {
    pub fn try_new(l: f64, c: f64, mu: f64) -> Result<Self, KernelError> {
        if !l.is_finite() || !c.is_finite() || !mu.is_finite() || l <= c || c <= 0.0 || mu <= 0.0 {
            return Err(KernelError::InvalidPowerLaw { l, c, mu });
        }
        Ok(Self {
            kind: KernelType::PowerLaw { l, c, mu },
            l_pow: l.powf(-mu),
            c_pow: c.powf(-mu),
        })
    }

    pub fn new(l: f64, c: f64, mu: f64) -> Self {
        Self::try_new(l, c, mu).expect("invalid power-law kernel configuration")
    }
}

impl Kernel for PowerLawKernel {
    fn sample_indexed(&self, key: RandomKey, sweep: u64, sample_index: u64) -> f64 {
        let (mu, c) = match self.kind {
            KernelType::PowerLaw { c, mu, .. } => (mu, c),
            _ => unreachable!("PowerLawKernel kind is fixed at construction"),
        };
        let u = unit_f64(key, sweep, DOMAIN_KERNEL_SAMPLE, sample_index, 0, 0);
        (u * (self.l_pow - self.c_pow) + self.c_pow)
            .powf(-1.0 / mu)
            .max(c)
    }

    fn kind(&self) -> KernelType {
        self.kind
    }

    fn boxed_clone(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct UniformKernel {
    kind: KernelType,
}

impl UniformKernel {
    pub fn try_new(l: f64, c: f64) -> Result<Self, KernelError> {
        if !l.is_finite() || !c.is_finite() || l <= c {
            return Err(KernelError::InvalidUniform { l, c });
        }
        Ok(Self {
            kind: KernelType::Uniform { l, c },
        })
    }

    pub fn new(l: f64, c: f64) -> Self {
        Self::try_new(l, c).expect("invalid uniform kernel configuration")
    }
}

impl Kernel for UniformKernel {
    fn sample_indexed(&self, key: RandomKey, sweep: u64, sample_index: u64) -> f64 {
        let (low, high) = match self.kind {
            KernelType::Uniform { l, c } => (c, l),
            _ => unreachable!("UniformKernel kind is fixed at construction"),
        };
        let u = unit_f64(key, sweep, DOMAIN_KERNEL_SAMPLE, sample_index, 0, 0);
        low + (high - low) * u
    }

    fn kind(&self) -> KernelType {
        self.kind
    }

    fn boxed_clone(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct NearestNeighborKernel {
    kind: KernelType,
    num_neighbors: usize,
}

impl NearestNeighborKernel {
    pub fn try_new(d: usize) -> Result<Self, KernelError> {
        let num_neighbors = d
            .checked_mul(2)
            .filter(|count| *count > 0)
            .ok_or(KernelError::InvalidNearestNeighborDimension { dimension: d })?;
        Ok(Self {
            kind: KernelType::NearestNeighbor { d },
            num_neighbors,
        })
    }

    pub fn new(d: usize) -> Self {
        Self::try_new(d).expect("invalid nearest-neighbor kernel configuration")
    }
}

impl Kernel for NearestNeighborKernel {
    fn sample_indexed(&self, key: RandomKey, sweep: u64, sample_index: u64) -> f64 {
        uniform_index(
            key,
            sweep,
            DOMAIN_KERNEL_SAMPLE,
            sample_index,
            0,
            self.num_neighbors,
        ) as f64
    }

    fn kind(&self) -> KernelType {
        self.kind
    }

    fn boxed_clone(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

/*!
Random displacement kernels for square lattices.

Purpose:
    A kernel describes the statistical rule used to choose a displacement for a
    square-lattice source site. `RandPairGenerator` uses these kernels to create
    nearest-neighbor steps or long-range moves. The kernel only samples the
    move length or nearest-neighbor direction code; the displacement module turns
    that sample into a vector, and the lattice representation later interprets
    raw target coordinates through its boundary condition.

Kernel families:
    - `NearestNeighbor`:
        Choose one of the `2 * rank` axis-aligned nearest-neighbor directions.
    - `Uniform`:
        Choose a continuous move length uniformly from `[c, l)`.
    - `PowerLaw`:
        Choose a continuous move length from a power-law distribution on
        `(c, l]` with exponent `mu`.

Parallelism:
    `sample(n)` uses Rayon to fill the output vector in parallel. Each output
    slot creates its own local random-number generator, so there is no shared
    mutable random state between worker threads.
*/

use rand::{RngExt, rng};
use rayon::prelude::*;
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub enum KernelType {
    PowerLaw { l: f64, c: f64, mu: f64 },
    Uniform { l: f64, c: f64 },
    NearestNeighbor { d: usize },
}

pub trait Kernel: Send + Sync {
    /// Sample `n` scalar values from this kernel's distribution.
    ///
    /// For `NearestNeighbor`, values are real direction codes in `[0, 2d)`.
    /// For `Uniform` and `PowerLaw`, values are move lengths.
    fn sample(&self, n: usize) -> Vec<f64>;

    /// Return the compact enum description of this kernel.
    fn kind(&self) -> KernelType;

    /// Clone this kernel behind a trait object.
    fn boxed_clone(&self) -> Box<dyn Kernel>;
}

impl Clone for Box<dyn Kernel> {
    #[inline]
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

#[inline]
pub fn create_kernel(kernel_type: KernelType) -> Box<dyn Kernel> {
    match kernel_type {
        KernelType::PowerLaw { l, c, mu } => Box::new(PowerLawKernel::new(l, c, mu)),
        KernelType::Uniform { l, c } => Box::new(UniformKernel::new(l, c)),
        KernelType::NearestNeighbor { d } => Box::new(NearestNeighborKernel::new(d)),
    }
}

#[derive(Debug, Clone)]
pub struct PowerLawKernel {
    kind: KernelType,
    l_pow: f64,
    c_pow: f64,
}

impl PowerLawKernel {
    #[inline]
    pub fn new(l: f64, c: f64, mu: f64) -> Self {
        assert!(l > c && c > 0.0, "PowerLawKernel: require l > c > 0");
        assert!(mu > 0.0, "PowerLawKernel: require mu > 0");
        let kind = KernelType::PowerLaw { l, c, mu };
        let l_pow = l.powf(-mu);
        let c_pow = c.powf(-mu);
        Self { kind, l_pow, c_pow }
    }
}

impl Kernel for PowerLawKernel {
    #[inline]
    fn sample(&self, n: usize) -> Vec<f64> {
        let (mu, l_pow, c_pow, c) = match self.kind {
            KernelType::PowerLaw { c, mu, .. } => (mu, self.l_pow, self.c_pow, c),
            _ => unreachable!("PowerLawKernel.kind must be PowerLaw"),
        };

        let mut out = vec![0.0_f64; n];
        out.par_iter_mut().for_each(|x| {
            let mut rng_local = rng();
            let mut u = rng_local.random::<f64>();
            if u >= 1.0 {
                u = f64::from_bits(1.0f64.to_bits() - 1);
            } else if u < 0.0 {
                u = 0.0;
            }
            *x = (u * (l_pow - c_pow) + c_pow).powf(-1.0 / mu).max(c);
        });
        out
    }

    #[inline]
    fn kind(&self) -> KernelType {
        self.kind
    }

    #[inline]
    fn boxed_clone(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct UniformKernel {
    kind: KernelType,
}

impl UniformKernel {
    #[inline]
    pub fn new(l: f64, c: f64) -> Self {
        assert!(l > c, "UniformKernel: require l > c");
        Self {
            kind: KernelType::Uniform { l, c },
        }
    }
}

impl Kernel for UniformKernel {
    #[inline]
    fn sample(&self, n: usize) -> Vec<f64> {
        let (low, high) = match self.kind {
            KernelType::Uniform { l, c } => (c, l),
            _ => unreachable!("UniformKernel.kind must be Uniform"),
        };
        let scale = high - low;

        let mut out = vec![0.0_f64; n];
        out.par_iter_mut().for_each(|x| {
            let mut rng_local = rng();
            *x = low + scale * rng_local.random::<f64>();
        });
        out
    }

    #[inline]
    fn kind(&self) -> KernelType {
        self.kind
    }

    #[inline]
    fn boxed_clone(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct NearestNeighborKernel {
    kind: KernelType,
    num_neighbors: f64,
}

impl NearestNeighborKernel {
    #[inline]
    pub fn new(d: usize) -> Self {
        assert!(d > 0, "NearestNeighborKernel: require d >= 1");
        let kind = KernelType::NearestNeighbor { d };
        let num_neighbors = (2 * d) as f64;
        Self {
            kind,
            num_neighbors,
        }
    }
}

impl Kernel for NearestNeighborKernel {
    #[inline]
    fn sample(&self, n: usize) -> Vec<f64> {
        match self.kind {
            KernelType::NearestNeighbor { .. } => {}
            _ => unreachable!("NearestNeighborKernel.kind must be NearestNeighbor"),
        }

        let high = self.num_neighbors;
        let mut out = vec![0.0_f64; n];
        out.par_iter_mut().for_each(|x| {
            let mut rng_local = rng();
            *x = high * rng_local.random::<f64>();
        });
        out
    }

    #[inline]
    fn kind(&self) -> KernelType {
        self.kind
    }

    #[inline]
    fn boxed_clone(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

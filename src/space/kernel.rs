// src/.../kernels.rs
/*!
Sampling **kernels** for generating random distances / neighbor picks.

What changed
------------
- No duplicated params in structs. Each kernel stores a single `kind: KernelType`
  (the source of truth). Any cached, *derived* values (e.g., `l_pow`, `c_pow`,
  `num_neighbors`) are kept separately and recomputed from `kind` in `new`.
- `Kernel::sample(n)` returns a `Vec<f64>` of length `n`.
- `Kernel::kind()` exposes the `KernelType` for introspection.

Concurrency
-----------
- Uses Rayon with per-element `rand::rng()`—no shared mutable state.
*/

use rayon::prelude::*;
use serde::Serialize;
use rand::{rng, RngExt};

// ======================================================================================
// ------------------------------------ Kernel Trait ------------------------------------
// ======================================================================================

#[derive(Debug, Clone, Copy, Serialize)]
pub enum KernelType {
    /// Power-law on distances with support `(c, l]`, exponent `μ > 0`.
    /// Inverse CDF: `x = (u * (l^{-μ} - c^{-μ}) + c^{-μ})^{-1/μ}`, `u ~ U[0,1)`.
    PowerLaw { l: f64, c: f64, mu: f64 },

    /// Continuous uniform on `[c, l)`.
    Uniform { l: f64, c: f64 },

    /// Uniform over **2d** nearest-neighbor directions (two per axis).
    /// Returns a **real** in `[0, 2d)`. Downstream may `floor()` to `{0, …, 2d-1}`.
    NearestNeighbor { d: usize },
}

pub trait Kernel: Send + Sync {
    /// Annotation:
    /// - Purpose: Executes `sample` logic for this module.
    /// - Parameters:
    ///   - `n` (`usize`): Parameter of type `usize` used by `sample`.
    fn sample(&self, n: usize) -> Vec<f64>;
    /// Annotation:
    /// - Purpose: Executes `kind` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn kind(&self) -> KernelType;
    /// Annotation:
    /// - Purpose: Executes `boxed_clone` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn boxed_clone(&self) -> Box<dyn Kernel>;
}

impl Clone for Box<dyn Kernel> {
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `clone` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn clone(&self) -> Self { self.boxed_clone() }
}

#[inline]
/// Annotation:
/// - Purpose: Executes `create_kernel` logic for this module.
/// - Parameters:
///   - `kernel_type` (`KernelType`): Parameter of type `KernelType` used by `create_kernel`.
pub fn create_kernel(kernel_type: KernelType) -> Box<dyn Kernel> {
    match kernel_type {
        KernelType::PowerLaw { l, c, mu } => Box::new(PowerLawKernel::new(l, c, mu)),
        KernelType::Uniform { l, c }      => Box::new(UniformKernel::new(l, c)),
        KernelType::NearestNeighbor { d } => Box::new(NearestNeighborKernel::new(d)),
    }
}

// ======================================================================================
// ---------------------------------- Concrete Kernels ----------------------------------
// ======================================================================================

/// Power-law `(c, l]` with exponent `μ > 0`.
#[derive(Debug, Clone)]
pub struct PowerLawKernel {
    kind: KernelType,  // single source of truth
    // cached, non-duplicative derived values:
    l_pow: f64,        // l^{-μ}
    c_pow: f64,        // c^{-μ}
}

impl PowerLawKernel {
    #[inline]
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `l` (`f64`): Parameter of type `f64` used by `new`.
    ///   - `c` (`f64`): Parameter of type `f64` used by `new`.
    ///   - `mu` (`f64`): Parameter of type `f64` used by `new`.
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
    /// Annotation:
    /// - Purpose: Executes `sample` logic for this module.
    /// - Parameters:
    ///   - `n` (`usize`): Parameter of type `usize` used by `sample`.
    fn sample(&self, n: usize) -> Vec<f64> {
        // Read params from `self.kind` (no duplicates).
        let (mu, l_pow, c_pow, c) = match self.kind {
            KernelType::PowerLaw { c, mu, .. } => (mu, self.l_pow, self.c_pow, c),
            _ => unreachable!("PowerLawKernel.kind must be PowerLaw"),
        };

        let mut out = vec![0.0_f64; n];
        out.par_iter_mut().for_each(|x| {
            let mut rng_local = rng();
            let mut u = rng_local.random::<f64>(); // [0,1)
            // clamp away from 1.0 exactly
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
    /// Annotation:
    /// - Purpose: Executes `kind` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn kind(&self) -> KernelType { self.kind }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `boxed_clone` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn boxed_clone(&self) -> Box<dyn Kernel> { Box::new(self.clone()) }
}

/// Continuous uniform on `[c, l)`.
#[derive(Debug, Clone)]
pub struct UniformKernel {
    kind: KernelType, // single source of truth
}

impl UniformKernel {
    #[inline]
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `l` (`f64`): Parameter of type `f64` used by `new`.
    ///   - `c` (`f64`): Parameter of type `f64` used by `new`.
    pub fn new(l: f64, c: f64) -> Self {
        assert!(l > c, "UniformKernel: require l > c");
        Self { kind: KernelType::Uniform { l, c } }
    }
}

impl Kernel for UniformKernel {
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `sample` logic for this module.
    /// - Parameters:
    ///   - `n` (`usize`): Parameter of type `usize` used by `sample`.
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
    /// Annotation:
    /// - Purpose: Executes `kind` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn kind(&self) -> KernelType { self.kind }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `boxed_clone` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn boxed_clone(&self) -> Box<dyn Kernel> { Box::new(self.clone()) }
}

/// Uniform over **2d** nearest-neighbor directions (returned as reals in `[0, 2d)`).
#[derive(Debug, Clone)]
pub struct NearestNeighborKernel {
    kind: KernelType,  // single source of truth
    // cached, non-duplicative derived value:
    num_neighbors: f64, // 2*d as f64
}

impl NearestNeighborKernel {
    #[inline]
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `d` (`usize`): Parameter of type `usize` used by `new`.
    pub fn new(d: usize) -> Self {
        assert!(d > 0, "NearestNeighborKernel: require d >= 1");
        let kind = KernelType::NearestNeighbor { d };
        let num_neighbors = (2 * d) as f64;
        Self { kind, num_neighbors }
    }
}

impl Kernel for NearestNeighborKernel {
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `sample` logic for this module.
    /// - Parameters:
    ///   - `n` (`usize`): Parameter of type `usize` used by `sample`.
    fn sample(&self, n: usize) -> Vec<f64> {
        // Pull `d` if you need it, but sampling only needs 2d as f64.
        let _d = match self.kind {
            KernelType::NearestNeighbor { d } => d,
            _ => unreachable!("NearestNeighborKernel.kind must be NearestNeighbor"),
        };
        let high = self.num_neighbors;

        let mut out = vec![0.0_f64; n];
        out.par_iter_mut().for_each(|x| {
            let mut rng_local = rng();
            *x = high * rng_local.random::<f64>(); // [0, 2d)
        });
        out
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `kind` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn kind(&self) -> KernelType { self.kind }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `boxed_clone` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn boxed_clone(&self) -> Box<dyn Kernel> { Box::new(self.clone()) }
}

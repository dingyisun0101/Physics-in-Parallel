use rayon::prelude::*;
use serde::Serialize;

use super::super::math::tensor::{ dense::Tensor, tensor_rand:: { fill_random_f64, RandType } };
use super::super::math::vector::{vector_list::VectorList};
use super::super::math::vector::vector_list_rand::fill_haar_vecs;
use super::discrete::GridConfig;

// ======================================================================================
// ------------------------------------ Kernel Trait ------------------------------------
// ======================================================================================
// This trait defines the interface for different kernel types used in sampling random distances or relations.
// And these values can be multiplied with any vector_list to generate displacement in the space.
// Each kernel must implement the `sample_n` method to generate samples and `boxed_clone`
// to allow cloning of the kernel trait object.
// The `Kernel` struct is designed to be thread-safe and can be used in parallel computations


#[derive(Debug, Clone, Copy, Serialize)]
pub enum KernelType {
    PowerLaw { l: f64, c: f64, mu: f64},
    Uniform {l: f64, c: f64},
    NearestNeighbor {d: usize}, // basis(d, i) denotes the ith neighbor
}

pub trait Kernel: Send + Sync {
    fn sample(&self, cache: &mut Tensor<f64>);
    fn boxed_clone(&self) -> Box<dyn Kernel>;
}

impl Clone for Box<dyn Kernel> {
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

pub fn create_kernel(kernel_type: KernelType) -> Box<dyn Kernel> {
    match kernel_type {
        KernelType::PowerLaw { mu , l, c} => {
            Box::new(PowerLawKernel::new(mu, l, c))
        }
        KernelType::Uniform { l, c } => {
            Box::new(UniformKernel::new(l, c))
        }
        KernelType::NearestNeighbor { d } => {
            Box::new(NearestNeighborKernel::new(d))
        }
    }
}




// ======================================================================================
// ---------------------------------- Types of Kernels ----------------------------------
// ======================================================================================

// -------------------------------- Power-Law Kernel -----------------------------------
// Power law Kernel is used to sample distances in a power-law distribution.
// It is defined by parameters l (scale), c (minimum distance), and mu (exponent).
// The sampling is done using the inverse transform sampling method.

#[derive(Debug, Clone)]
pub struct PowerLawKernel {
    pub l: f64,
    pub c: f64,
    pub mu: f64,
    pub l_pow: f64,
    pub c_pow: f64,
}

impl PowerLawKernel {
    pub fn new(l: f64, c: f64, mu: f64) -> Self {
        let l_pow = l.powf(-mu);
        let c_pow = c.powf(-mu);
        Self { l, c, mu, l_pow, c_pow }
    }
}

impl Kernel for PowerLawKernel {
    fn sample(&self, cache: &mut Tensor<f64>) {
        let mu    = self.mu;
        let l_pow = self.l_pow; // l^-mu
        let c_pow = self.c_pow; // c^-mu

        // 1) Fill with U ~ Uniform[0,1)
        fill_random_f64(cache, RandType::Uniform { low: 0.0, high: 1.0 });

        // 2) In-place inverse-CDF transform:
        //    x = (u*(l^-mu - c^-mu) + c^-mu)^(-1/mu)
        cache
            .data
            .par_iter_mut()
            .for_each(|u| {
                // clamp to [0,1) to guard against 1.0
                let uu = if *u >= 1.0 {
                    // largest f64 < 1.0
                    f64::from_bits(0x3fefffff_fffffff)
                } else if *u < 0.0 {
                    0.0
                } else {
                    *u
                };
                *u = (uu * (l_pow - c_pow) + c_pow).powf(-1.0 / mu);
            });
    }

    fn boxed_clone(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

// ------------------------------------ Uniform Kernel ------------------------------------
#[derive(Debug, Clone)]
pub struct UniformKernel {
    pub l: f64,
    pub c: f64,
}

impl UniformKernel {
    pub fn new(l: f64, c: f64) -> Self {
        assert!(l > c, "l must be greater than c");
        Self { l, c }
    }
}

impl Kernel for UniformKernel {
    fn sample(&self, cache: &mut Tensor<f64>) {
        // Fill with U ~ Uniform[c,l)
        fill_random_f64(cache, RandType::Uniform { low: self.c, high: self.l });
    }

    fn boxed_clone(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

// ------------------------------------ Nearest-Neighbor Kernel ------------------------------------
#[derive(Debug, Clone)]
pub struct NearestNeighborKernel {
    pub d: usize,
    pub num_neighbors: f64,
}

impl NearestNeighborKernel {
    pub fn new(d: usize) -> Self {
        let num_neighbors = (2 * d) as f64;
        Self {d, num_neighbors}
    }
}

impl Kernel for NearestNeighborKernel {
    fn sample(&self, cache: &mut Tensor<f64>) {
        // Fill with U ~ Uniform[c,l)
        fill_random_f64(cache, RandType::Uniform { low: 0.0, high:  self.num_neighbors});
    }

    fn boxed_clone(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

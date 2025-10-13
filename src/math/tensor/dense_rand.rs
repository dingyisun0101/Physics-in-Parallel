// src/math_foundations/tensor/tensor_rand.rs
/*!
Random filling with **per-slice RNGs**.

- Split the flat buffer into `min(num_rngs, n)` contiguous slices.
- One `SmallRng` per slice; optional deterministic seeding.
- Call `refresh(&mut tensor)` to (re)fill any `Tensor<T>` in-place.
- Supports: `f64` (Uniform, Normal, Bernoulli), `i64` (UniformInt, Bernoulli),
            `usize` (UniformInt), `isize` (UniformInt).

Tune `num_rngs` to cores/cache. Default: `NUM_RNGS = 32`.
*/

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Bernoulli, Distribution, Normal, Uniform};
use rayon::prelude::*;

use crate::math::tensor::dense::Tensor;
use crate::math::scalar::Scalar;

//===================================================================
// ---------------------------- Config ------------------------------
//===================================================================

pub const NUM_RNGS: usize = 32; // default number of RNGs

//===================================================================
// -------------------------- Basic Types ---------------------------
//===================================================================

#[derive(Debug, Clone)]
pub enum RandType {
    Uniform { low: f64, high: f64 },      // floats: [low, high)
    UniformInt { low: i64, high: i64 },   // ints:   [low, high]
    Normal { mean: f64, std: f64 },
    Bernoulli { p: f64 },
}

#[derive(Debug, Clone)]
pub struct TensorRandFiller {
    kind: RandType,
    num_rngs: usize,
    rngs: Vec<SmallRng>, // prebuilt pool; we slice to the active count per refresh
}

impl TensorRandFiller {
    #[inline]
    pub fn new(kind: RandType, num_rngs: Option<usize>) -> Self {
        let req = match num_rngs {
            Some(0) => panic!("num_rngs must be > 0"),
            Some(n) => n,
            None => NUM_RNGS,
        };
        let mut master = SmallRng::from_os_rng();
        let mut rngs: Vec<SmallRng> = (0..req)
            .map(|_| SmallRng::try_from_rng(&mut master).expect("Failed to seed RNG"))
            .collect();
        rngs.shrink_to_fit();
        Self { kind, num_rngs: req, rngs }
    }

    #[inline]
    pub fn new_with_seed(kind: RandType, num_rngs: Option<usize>, seed: u64) -> Self {
        let req = match num_rngs {
            Some(0) => panic!("num_rngs must be > 0"),
            Some(n) => n,
            None => NUM_RNGS,
        };
        let mut master = SmallRng::seed_from_u64(seed);
        let mut rngs: Vec<SmallRng> = (0..req)
            .map(|_| SmallRng::try_from_rng(&mut master).expect("Failed to seed RNG"))
            .collect();
        rngs.shrink_to_fit();
        Self { kind, num_rngs: req, rngs }
    }

    #[inline]
    fn active_slices(&self, n: usize) -> usize {
        if n == 0 { 0 } else { self.num_rngs.min(n) }
    }

    #[inline]
    fn chunk_len(&self, n: usize, slices: usize) -> usize {
        if n == 0 || slices == 0 { 0 } else { (n + slices - 1) / slices } // ceil(n/slices)
    }

    /// Generic refresh: dispatches to the right impl based on `T`.
    #[inline]
    pub fn refresh<T: TensorRandElement>(&mut self, tensor: &mut Tensor<T>) {
        T::fill(self, tensor);
    }

    /// Access/modify the distribution kind if you want to reuse the filler.
    #[inline] pub fn kind(&self) -> &RandType { &self.kind }
    #[inline] pub fn set_kind(&mut self, k: RandType) { self.kind = k; }
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
    fn fill(f: &mut TensorRandFiller, tensor: &mut Tensor<Self>);
}

// ---------------------------- f64 ---------------------------------
impl TensorRandElement for f64 {
    #[inline]
    fn fill(f: &mut TensorRandFiller, tensor: &mut Tensor<f64>) {
        let n = tensor.data.len();
        if n == 0 { return; }

        let slices = f.active_slices(n);
        let chunk_len = f.chunk_len(n, slices);
        let expected = (n + chunk_len - 1) / chunk_len;
        assert_eq!(expected, slices);

        let rngs = &mut f.rngs[..slices];

        match f.kind {
            RandType::Uniform { low, high } => {
                let _ = Uniform::new(low, high)
                    .map_err(|_| panic!("Uniform<f64>: low must be < high"));
                tensor.data
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| {
                        let dist = Uniform::new(low, high).unwrap();
                        for x in chunk { *x = dist.sample(rng); }
                    });
            }
            RandType::Normal { mean, std } => {
                let _ = Normal::new(mean, std)
                    .map_err(|_| panic!("Normal<f64>: std must be > 0"));
                tensor.data
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| {
                        let dist = Normal::new(mean, std).unwrap();
                        for x in chunk { *x = dist.sample(rng); }
                    });
            }
            RandType::Bernoulli { p } => {
                let _ = Bernoulli::new(p)
                    .map_err(|_| panic!("Bernoulli<f64>: p must be in [0,1]"));
                tensor.data
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| {
                        let dist = Bernoulli::new(p).unwrap();
                        for x in chunk { *x = if dist.sample(rng) { 1.0 } else { 0.0 }; }
                    });
            }
            RandType::UniformInt { .. } => {
                panic!("Tensor<f64>: UniformInt not supported; use RandType::Uniform");
            }
        }
    }
}

// ---------------------------- i64 ---------------------------------
impl TensorRandElement for i64 {
    #[inline]
    fn fill(f: &mut TensorRandFiller, tensor: &mut Tensor<i64>) {
        let n = tensor.data.len();
        if n == 0 { return; }

        let slices = f.active_slices(n);
        let chunk_len = f.chunk_len(n, slices);
        let expected = (n + chunk_len - 1) / chunk_len;
        assert_eq!(expected, slices);

        let rngs = &mut f.rngs[..slices];

        match f.kind {
            RandType::UniformInt { low, high } => {
                let _ = Uniform::new_inclusive(low, high)
                    .map_err(|_| panic!("UniformInt<i64>: low must be <= high"));
                tensor.data
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| {
                        let dist = Uniform::new_inclusive(low, high).unwrap();
                        for x in chunk { *x = dist.sample(rng); }
                    });
            }
            RandType::Bernoulli { p } => {
                let _ = Bernoulli::new(p)
                    .map_err(|_| panic!("Bernoulli<i64>: p must be in [0,1]"));
                tensor.data
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| {
                        let dist = Bernoulli::new(p).unwrap();
                        for x in chunk { *x = if dist.sample(rng) { 1 } else { 0 }; }
                    });
            }
            _ => panic!("Tensor<i64>: supported kinds are UniformInt and Bernoulli"),
        }
    }
}

// ---------------------------- usize -------------------------------
impl TensorRandElement for usize {
    #[inline]
    fn fill(f: &mut TensorRandFiller, tensor: &mut Tensor<usize>) {
        let n = tensor.data.len();
        if n == 0 { return; }

        let slices = f.active_slices(n);
        let chunk_len = f.chunk_len(n, slices);
        let expected = (n + chunk_len - 1) / chunk_len;
        assert_eq!(expected, slices);

        let rngs = &mut f.rngs[..slices];

        match f.kind {
            RandType::UniformInt { low, high } => {
                let (low_u, high_u) = match (usize::try_from(low), usize::try_from(high)) {
                    (Ok(lo), Ok(hi)) if lo <= hi => (lo, hi),
                    _ => panic!("UniformInt<usize>: invalid bounds (low={low}, high={high})"),
                };
                let _ = Uniform::new_inclusive(low_u, high_u)
                    .map_err(|_| panic!("UniformInt<usize>: low must be <= high"));
                tensor.data
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| {
                        let dist = Uniform::new_inclusive(low_u, high_u).unwrap();
                        for x in chunk { *x = dist.sample(rng); }
                    });
            }
            _ => panic!("Tensor<usize>: only UniformInt is supported"),
        }
    }
}

// ---------------------------- isize -------------------------------
// Use i64 inclusive uniform and cast after validating bounds.
impl TensorRandElement for isize {
    #[inline]
    fn fill(f: &mut TensorRandFiller, tensor: &mut Tensor<isize>) {
        let n = tensor.data.len();
        if n == 0 { return; }

        let slices = f.active_slices(n);
        let chunk_len = f.chunk_len(n, slices);
        let expected = (n + chunk_len - 1) / chunk_len;
        assert_eq!(expected, slices, "RNG pool vs slice mismatch");

        let rngs = &mut f.rngs[..slices];

        match f.kind {
            RandType::UniformInt { low, high } => {
                let _ = isize::try_from(low)
                    .map_err(|_| panic!("UniformInt<isize>: low out of isize range"));
                let _ = isize::try_from(high)
                    .map_err(|_| panic!("UniformInt<isize>: high out of isize range"));
                let _ = Uniform::new_inclusive(low, high)
                    .map_err(|_| panic!("UniformInt<isize>: low must be <= high"));

                tensor.data
                    .par_chunks_mut(chunk_len)
                    .zip(rngs.par_iter_mut())
                    .for_each(|(chunk, rng)| {
                        let dist_i64 = Uniform::<i64>::new_inclusive(low, high).unwrap();
                        for x in chunk {
                            let v = dist_i64.sample(rng);
                            *x = v as isize; // safe after bound checks
                        }
                    });
            }
            _ => panic!("Tensor<isize>: only UniformInt is supported"),
        }
    }
}
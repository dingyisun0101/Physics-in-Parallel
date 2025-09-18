/*
    Fill **dense** tensors with random numbers (parallelized).

    Design goals
    ------------
    - **Thread-safe & parallel**: Every element is sampled independently in a
      `par_iter_mut()` loop. Each loop body creates its own local RNG via
      `rand::rng()` so there is no shared mutable state or locking.
    - **Clear type split**:
        * `Tensor<f64>`: Uniform / Normal / Bernoulli (0.0 or 1.0)
        * `Tensor<i64>`: UniformInt / Bernoulli (0 or 1)
        * `Tensor<usize>`: UniformInt
    - **No duplication**: Small distribution-specific fillers; user code calls
      the single public dispatcher `fill_random`.

    Notes
    -----
    - Bounds/parameter validation is performed by distribution constructors.
    - `rand::rng()` (Rand ≥ 0.9) yields a fast thread-local RNG; great with Rayon.
*/

use rayon::prelude::*;
use rand::rng;
use rand_distr::{Bernoulli, Distribution, Normal, Uniform};

use super::super::scalar::Scalar;
use super::dense::Tensor;

// ============================================================================
// ------------------------- Distribution-specific fillers --------------------
// ============================================================================

#[inline]
fn fill_normal_f64(tensor: &mut Tensor<f64>, mean: f64, std: f64) {
    let dist = Normal::new(mean, std)
        .expect("fill_normal_f64: invalid normal params (std must be > 0)");
    tensor.data.par_iter_mut().for_each(|x| {
        let mut rng_local = rng();
        *x = dist.sample(&mut rng_local);
    });
}

#[inline]
fn fill_uniform_f64(tensor: &mut Tensor<f64>, low: f64, high: f64) {
    let dist = Uniform::new(low, high)
        .expect("fill_uniform_f64: invalid uniform bounds (require low < high)");
    tensor.data.par_iter_mut().for_each(|x| {
        let mut rng_local = rng();
        *x = dist.sample(&mut rng_local);
    });
}

#[inline]
fn fill_bernoulli_f64(tensor: &mut Tensor<f64>, p: f64) {
    let dist = Bernoulli::new(p)
        .expect("fill_bernoulli_f64: invalid p (must be in [0,1])");
    tensor.data.par_iter_mut().for_each(|x| {
        let mut rng_local = rng();
        *x = if dist.sample(&mut rng_local) { 1.0 } else { 0.0 };
    });
}

#[inline]
fn fill_uniform_i64(tensor: &mut Tensor<i64>, low: i64, high: i64) {
    // For integers, `Uniform::new_inclusive` constructs directly and panics if low > high.
    let dist = Uniform::new_inclusive(low, high).unwrap();
    tensor.data.par_iter_mut().for_each(|x| {
        let mut rng_local = rng();
        *x = dist.sample(&mut rng_local);
    });
}

#[inline]
fn fill_bernoulli_i64(tensor: &mut Tensor<i64>, p: f64) {
    let dist = Bernoulli::new(p)
        .expect("fill_bernoulli_i64: invalid p (must be in [0,1])");
    tensor.data.par_iter_mut().for_each(|x| {
        let mut rng_local = rng();
        *x = if dist.sample(&mut rng_local) { 1 } else { 0 };
    });
}

/// `usize` version
#[inline]
pub fn fill_uniform_usize(tensor: &mut Tensor<usize>, low: i64, high: i64) {
    let (low_u, high_u) = match (usize::try_from(low), usize::try_from(high)) {
        (Ok(lo), Ok(hi)) if lo <= hi => (lo, hi),
        _ => panic!(
            "fill_uniform_usize: invalid bounds for usize: low={low}, high={high}"
        ),
    };
    let dist = Uniform::new_inclusive(low_u, high_u).unwrap();
    tensor.data.par_iter_mut().for_each(|x| {
        let mut rng_local = rng();
        *x = dist.sample(&mut rng_local);
    });
}

/// `isize` version
#[inline]
pub fn fill_uniform_isize(tensor: &mut Tensor<isize>, low: i64, high: i64) {
    let dist = Uniform::new_inclusive(low, high).unwrap();
    tensor.data.par_iter_mut().for_each(|x| {
        let mut rng_local = rng();
        *x = dist.sample(&mut rng_local) as isize;
    });
}

// ============================================================================
// ------------------------------ Unified dispatcher --------------------------
// ============================================================================

/// High-level random specification for the dispatcher.
#[derive(Debug, Clone)]
pub enum RandType {
    /// Continuous uniform on **[low, high)** for floating-point tensors.
    Uniform { low: f64, high: f64 },
    /// Integer uniform on **[low, high]** inclusive for integer tensors.
    UniformInt { low: i64, high: i64 },
    /// Normal (Gaussian) with mean/std for floating-point tensors.
    Normal { mean: f64, std: f64 },
    /// Bernoulli with success prob `p ∈ [0,1]`.
    Bernoulli { p: f64 },
}

/// Fill a `Tensor<T>` with random values according to `spec`.
///
/// Supported:
/// - **f64**: `Uniform`, `Normal`, `Bernoulli`
/// - **i64**: `UniformInt`, `Bernoulli`
/// - **usize**: `UniformInt`
///
/// Panics on unsupported combinations.
#[inline]
pub fn fill_random<T>(tensor: &mut Tensor<T>, spec: RandType)
where
    T: FillDispatch, // FillDispatch already implies Scalar below
{
    T::dispatch_fill(tensor, spec)
}

// Type-level dispatch: implement per element type.
pub trait FillDispatch: Scalar + Sized {
    fn dispatch_fill(tensor: &mut Tensor<Self>, spec: RandType);
}

// ---------------------- impl for f64 ----------------------
impl FillDispatch for f64 {
    fn dispatch_fill(tensor: &mut Tensor<Self>, spec: RandType) {
        match spec {
            RandType::Uniform { low, high } => fill_uniform_f64(tensor, low, high),
            RandType::Normal { mean, std } => fill_normal_f64(tensor, mean, std),
            RandType::Bernoulli { p } => fill_bernoulli_f64(tensor, p),
            RandType::UniformInt { .. } => {
                panic!("fill_random<f64>: UniformInt not supported (use Uniform)")
            }
        }
    }
}

// ---------------------- impl for i64 ----------------------
impl FillDispatch for i64 {
    fn dispatch_fill(tensor: &mut Tensor<Self>, spec: RandType) {
        match spec {
            RandType::UniformInt { low, high } => fill_uniform_i64(tensor, low, high),
            RandType::Bernoulli { p } => fill_bernoulli_i64(tensor, p),
            _ => panic!("fill_random<i64>: only UniformInt and Bernoulli supported"),
        }
    }
}

// ---------------------- impl for usize ----------------------
impl FillDispatch for usize {
    fn dispatch_fill(tensor: &mut Tensor<Self>, spec: RandType) {
        match spec {
            RandType::UniformInt { low, high } => fill_uniform_usize(tensor, low, high),
            RandType::Bernoulli { .. } => {
                panic!("fill_random<usize>: Bernoulli not supported")
            }
            _ => panic!("fill_random<usize>: only UniformInt supported"),
        }
    }
}

// ---------------------- impl for isize ----------------------
impl FillDispatch for isize {
    fn dispatch_fill(tensor: &mut Tensor<Self>, spec: RandType) {
        match spec {
            RandType::UniformInt { low, high } => fill_uniform_isize(tensor, low, high),
            RandType::Bernoulli { .. } => {
                panic!("fill_random<usize>: Bernoulli not supported")
            }
            _ => panic!("fill_random<usize>: only UniformInt supported"),
        }
    }
}

/* ------------------------------ Examples ------------------------------------

let mut t1: Tensor<f64>  = Tensor::zeros(vec![100]);
fill_random(&mut t1, RandType::Normal { mean: 0.0, std: 1.0 });

let mut t2: Tensor<i64>  = Tensor::zeros(vec![100]);
fill_random(&mut t2, RandType::UniformInt { low: -5, high: 5 });

let mut t3: Tensor<usize> = Tensor::zeros(vec![100]);
fill_random(&mut t3, RandType::UniformInt { low: 0, high: 10 });

------------------------------------------------------------------------------- */

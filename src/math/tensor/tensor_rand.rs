// src/math_foundations/tensor/tensor_rand.rs
/*
    Fill tensors (dense **or** sparse) with random numbers (parallelized).

    Design goals
    ------------
    - **Thread-safe & parallel**: Every element is sampled independently via
      `TensorTrait::par_map_inplace`, using a fresh thread-local RNG (`rand::rng()`).
    - **Backend-agnostic**: Works with any tensor that implements the `Tensor` trait
      (dense or sparse), applying maps **in-place** over elements (sparse: only
      existing nonzeros are touched by `par_map_inplace` per your trait semantics).
    - **Clear type split**:
        * `T = f64`: Uniform / Normal / Bernoulli (0.0 or 1.0)
        * `T = i64`: UniformInt / Bernoulli (0 or 1)
        * `T = usize`: UniformInt
        * `T = isize`: UniformInt
    - **No duplication**: Small distribution-specific fillers; user code calls
      the single public dispatcher `fill_random`.

    Notes
    -----
    - Bounds/parameter validation is performed by distribution constructors.
    - `rand::rng()` (Rand ≥ 0.9) yields a fast thread-local RNG; great with Rayon.
*/

use rand::rng;
use rand_distr::{Bernoulli, Distribution, Normal, Uniform};

use super::super::scalar::Scalar;
use super::tensor_trait::TensorTrait;

// ============================================================================
// ------------------------- Distribution-specific fillers --------------------
// ============================================================================

#[inline]
pub fn fill_normal_f64<Ten>(tensor: &mut Ten, mean: f64, std: f64)
where
    Ten: TensorTrait<f64>,
{
    let dist = Normal::new(mean, std)
        .expect("fill_normal_f64: invalid normal params (std must be > 0)");

    tensor.par_map_in_place(|_| {
        let mut rng_local = rng();
        dist.sample(&mut rng_local)
    });
}

#[inline]
pub fn fill_uniform_f64<Ten>(tensor: &mut Ten, low: f64, high: f64)
where
    Ten: TensorTrait<f64>,
{
    // For floats, Uniform::new expects low < high.
    let dist = Uniform::new(low, high).expect("fill_uniform_f64: low must be < high");
    tensor.par_map_in_place(|_| {
        let mut rng_local = rng();
        dist.sample(&mut rng_local)
    });
}

#[inline]
pub fn fill_bernoulli_f64<Ten>(tensor: &mut Ten, p: f64)
where
    Ten: TensorTrait<f64>,
{
    let dist =
        Bernoulli::new(p).expect("fill_bernoulli_f64: invalid p (must be in [0,1])");
    tensor.par_map_in_place(|_| {
        let mut rng_local = rng();
        if dist.sample(&mut rng_local) {
            1.0
        } else {
            0.0
        }
    });
}

#[inline]
pub fn fill_uniform_i64<Ten>(tensor: &mut Ten, low: i64, high: i64)
where
    Ten: TensorTrait<i64>,
{
    let dist = Uniform::new_inclusive(low, high)
        .expect("fill_uniform_i64: low must be <= high");
    tensor.par_map_in_place(|_| {
        let mut rng_local = rng();
        dist.sample(&mut rng_local)
    });
}

#[inline]
pub fn fill_bernoulli_i64<Ten>(tensor: &mut Ten, p: f64)
where
    Ten: TensorTrait<i64>,
{
    let dist =
        Bernoulli::new(p).expect("fill_bernoulli_i64: invalid p (must be in [0,1])");
    tensor.par_map_in_place(|_| {
        let mut rng_local = rng();
        if dist.sample(&mut rng_local) {
            1
        } else {
            0
        }
    });
}

/// `usize` version
#[inline]
pub fn fill_uniform_usize<Ten>(tensor: &mut Ten, low: i64, high: i64)
where
    Ten: TensorTrait<usize>,
{
    let (low_u, high_u) = match (usize::try_from(low), usize::try_from(high)) {
        (Ok(lo), Ok(hi)) if lo <= hi => (lo, hi),
        _ => panic!("fill_uniform_usize: invalid bounds for usize: low={low}, high={high}"),
    };
    let dist = Uniform::new_inclusive(low_u, high_u)
        .expect("fill_uniform_usize: low must be <= high (usize)");
    tensor.par_map_in_place(|_| {
        let mut rng_local = rng();
        dist.sample(&mut rng_local)
    });
}

/// `isize` version
#[inline]
pub fn fill_uniform_isize<Ten>(tensor: &mut Ten, low: i64, high: i64)
where
    Ten: TensorTrait<isize>,
{
    let dist = Uniform::new_inclusive(low, high)
        .expect("fill_uniform_isize: low must be <= high");
    tensor.par_map_in_place(|_| {
        let mut rng_local = rng();
        dist.sample(&mut rng_local) as isize
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

/// Fill a tensor with random values according to `spec`.
///
/// Supported element types:
/// - **f64**: `Uniform`, `Normal`, `Bernoulli`
/// - **i64**: `UniformInt`, `Bernoulli`
/// - **usize**: `UniformInt`
/// - **isize**: `UniformInt`
///
/// Panics on unsupported combinations.
#[inline]
pub fn fill_random<T, Ten>(tensor: &mut Ten, spec: RandType)
where
    T: FillDispatch + Copy + Send + Sync,
    Ten: TensorTrait<T>,
{
    T::dispatch_fill(tensor, spec)
}

// Type-level dispatch: implement per element type.
pub trait FillDispatch: Scalar + Sized {
    fn dispatch_fill<Ten>(tensor: &mut Ten, spec: RandType)
    where
        Ten: TensorTrait<Self>;
}

// ---------------------- impl for f64 ----------------------
impl FillDispatch for f64 {
    fn dispatch_fill<Ten>(tensor: &mut Ten, spec: RandType)
    where
        Ten: TensorTrait<Self>,
    {
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
    fn dispatch_fill<Ten>(tensor: &mut Ten, spec: RandType)
    where
        Ten: TensorTrait<Self>,
    {
        match spec {
            RandType::UniformInt { low, high } => fill_uniform_i64(tensor, low, high),
            RandType::Bernoulli { p } => fill_bernoulli_i64(tensor, p),
            _ => panic!("fill_random<i64>: only UniformInt and Bernoulli supported"),
        }
    }
}

// ---------------------- impl for usize ----------------------
impl FillDispatch for usize {
    fn dispatch_fill<Ten>(tensor: &mut Ten, spec: RandType)
    where
        Ten: TensorTrait<Self>,
    {
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
    fn dispatch_fill<Ten>(tensor: &mut Ten, spec: RandType)
    where
        Ten: TensorTrait<Self>,
    {
        match spec {
            RandType::UniformInt { low, high } => fill_uniform_isize(tensor, low, high),
            RandType::Bernoulli { .. } => {
                panic!("fill_random<isize>: Bernoulli not supported")
            }
            _ => panic!("fill_random<isize>: only UniformInt supported"),
        }
    }
}

/* ------------------------------ Examples ------------------------------------

use crate::math_foundations::tensor::tensor_rand::{fill_random, RandType};
use crate::math_foundations::tensor::tensor_trait::Tensor as TensorTrait;

// Works with any backend that implements the Tensor trait (dense or sparse):
fn demo_fill_any_backend<T, Ten>(t: &mut Ten)
where
    T: FillDispatch + Copy + Send + Sync,
    Ten: TensorTrait<T>,
{
    fill_random::<T, _>(t, RandType::UniformInt { low: 0, high: 10 });
}

------------------------------------------------------------------------------- */

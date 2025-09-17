/*
    Fill dense tensors with random numbers 
*/

use rayon::prelude::*;
use rand::rng;
use rand_distr::{Distribution, Normal, Uniform, Bernoulli};
use super::dense::Tensor;


// ============================================================================
// ----------------------------- Types of RNG ---------------------------------
// ============================================================================

#[derive(Debug, Clone)]
pub enum RandType {
    // Continuous uniform on [low, high)
    Uniform { low: f64, high: f64 },

    // Integer uniform on [low, high] inclusive
    UniformInt { low: i64, high: i64 },
    
    // Normal with mean/std
    Normal  { mean: f64, std: f64 },
    
    // Bernoulli with success prob p
    Bernoulli { p: f64 },
}


// ============================================================================
// ------------------------- Base Type Rand Fillers  --------------------------
// ============================================================================

/// Normal N(mean, std)
fn fill_normal_f64(tensor: &mut Tensor<f64>, mean: f64, std: f64) {
    let dist = Normal::new(mean, std).expect("invalid normal params");
    tensor.data.par_iter_mut().for_each(|x| {
        let mut rng_local = rng();
        *x = dist.sample(&mut rng_local);
    });
}

/// Uniform(low, high)
fn fill_uniform_f64(tensor: &mut Tensor<f64>, low: f64, high: f64) {
    let dist = Uniform::new(low, high).expect("invalid uniform bounds");
    tensor.data.par_iter_mut().for_each(|x| {
        let mut rng_local = rng();
        *x = dist.sample(&mut rng_local);
    });
}

/// Integer uniform inclusive [low, high]
fn fill_uniform_i64(tensor: &mut Tensor<i64>, low: i64, high: i64) {
    let dist = Uniform::new_inclusive(low, high).expect("invalid int uniform bounds");
    tensor.data.par_iter_mut().for_each(|x| {
        let mut rng_local = rng();
        *x = dist.sample(&mut rng_local);
    });
}

/// Bernoulli(p) mapped to {0,1}
fn fill_bernoulli_i64(tensor: &mut Tensor<i64>, p: f64) {
    let dist = Bernoulli::new(p).expect("invalid bernoulli p");
    tensor.data.par_iter_mut().for_each(|x| {
        let mut rng_local = rng();
        *x = if dist.sample(&mut rng_local) { 1 } else { 0 };
    });
}





// ============================================================================
// ------------------------- Base Type Dispatchers  ---------------------------
// ============================================================================

/// Dispatcher for f64
pub fn fill_random_f64(tensor: &mut Tensor<f64>, spec: RandType) {
    match spec {
        RandType::Uniform { low, high } => fill_uniform_f64(tensor, low, high),
        RandType::Normal { mean, std } => fill_normal_f64(tensor, mean, std),
        _ => { panic!("fill_random_f64: Int is not supported for f64 tensors");}
    }
}

/// Dispatcher for i64
pub fn fill_random_i64(tensor: &mut Tensor<i64>, spec: RandType) {
    match spec {
        RandType::UniformInt { low, high } => fill_uniform_i64(tensor, low, high),
        RandType::Bernoulli {p} => fill_bernoulli_i64(tensor, p),
        _ => { panic!("fill_random_i64: Float is not supported for i64 tensors");}
    }
}
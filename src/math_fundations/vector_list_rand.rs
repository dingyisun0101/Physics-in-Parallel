// src/math_foundations/vector_list_rand.rs
/* 
    Random fillers & utilities that work **in-place** on `VectorList<T, D>`
*/

use rayon::prelude::*;
use rand_distr::{Distribution, StandardNormal, Uniform};
use super::vector_list::VectorList;

// f64 fillers
pub fn fill_normal<const D: usize>(vl: &mut VectorList<f64, D>) {
    for k in 0..D {
        let col = vl.dim_slice_mut(k);
        col.par_iter_mut().for_each(|x| {
            let mut rng_local = rand::rng();
            *x = StandardNormal.sample(&mut rng_local);
        });
    }
}

pub fn fill_uniform_f64<const D: usize>(vl: &mut VectorList<f64, D>, a: f64, b: f64) {
    let dist = Uniform::new(a, b).expect("invalid uniform bounds");
    for k in 0..D {
        let col = vl.dim_slice_mut(k);
        col.par_iter_mut().for_each(|x| {
            let mut rng_local = rand::rng();
            *x = dist.sample(&mut rng_local);
        });
    }
}

//usize filler
pub fn fill_uniform_usize<const D: usize>(vl: &mut VectorList<usize, D>, a: usize, b: usize) {
    let dist = Uniform::new(a, b).expect("invalid uniform bounds");
    for k in 0..D {
        let col = vl.dim_slice_mut(k);
        col.par_iter_mut().for_each(|x| {
            let mut rng_local = rand::rng();
            *x = dist.sample(&mut rng_local);
        });
    }
}
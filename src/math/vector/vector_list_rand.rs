// src/math_foundations/vector_list_rand.rs
/* 
    Random fillers & utilities that work **in-place** on `VectorList<T, D>`
*/
use rand::seq::SliceRandom;
use rand::rng;
use rayon::prelude::*;

use super::super::scalar::Scalar;
use super::super::tensor::tensor_rand::{RandType, fill_random_f64, fill_random_i64};
use super::vector_list::VectorList;



// Fill a VectorList<f64, D> with Haar-random unit vectors (uniform on S^{D-1}).
// Implementation: i.i.d. Normal(0,1) for every entry, then per-vector normalization.
pub fn fill_haar_vecs<const D: usize>(vl: &mut VectorList<f64, D>) {
    // 1) i.i.d. Gaussian entries
    let rand_type = RandType::Normal { mean: (0.0), std: (1.0) };
    fill_random_f64(&mut vl.data, rand_type);
    // 2) in-place L2 normalization per vector (uses your `normalize()` impl)
    vl.normalize();
}


// Permute the sequence of vectors in-place (parallel across dimensions).
// Applies one global permutation to all D rows of the SoA [D, n].
// WARNING: very expensive. Try shuffling the indices only. 
pub fn permute_vector_list<T: Scalar, const D: usize>(
    vl: &mut VectorList<T, D>,
) {
    let n = vl.num_vec();
    if n <= 1 { return; }

    // 1) Build a random permutation
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng());

    // 2) Clone source; write results back into vl.data.data
    let src = vl.data.data.clone();

    // 3) Parallel over rows (each row is one dimension slice of length n)
    vl.data
        .data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(k, row_dst)| {
            let row_src = &src[k * n .. (k + 1) * n];
            // Apply the same permutation to this row
            for (j, &i_src) in indices.iter().enumerate() {
                row_dst[j] = row_src[i_src];
            }
        });
}


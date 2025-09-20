// src/math_foundations/vector_list_rand.rs
/*!
Random fillers & utilities that work **in-place** on `VectorList<T>`.

Design goals
------------
- **Zero shared mutable state** across threads (Rayon-friendly).
- **Reuse** existing building blocks (`tensor_rand` + `VectorList` methods).
- **Deterministic shape semantics** for all `D ≥ 1` with SoA layout `[D, n]`.
- **Permutation is global per-vector**: the same permutation is applied to all
  D rows so vectors remain intact while being reordered.

Notes
-----
- `fill_haar_vectors` draws i.i.d. `Normal(0,1)` for every entry and then
  L2-normalizes **each vector** → Haar (uniform) on the unit sphere `S^{D-1}`.
- The normalization step calls your `VectorList`’s `normalize`.
- `permute_vector_list` builds one random permutation of `0..n` and applies it
  to every dimension slice (row). This is **O(D·n)** with an extra clone of the
  underlying data to avoid aliasing while writing in place.
*/

use rand::rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use super::super::scalar::Scalar;
use super::super::tensor::{
    tensor_rand::{fill_random, RandType},
};
use super::vector_list::VectorList;

// ============================================================================
// -------------------------- Haar-random unit vectors ------------------------
// ============================================================================

/// Fill a `VectorList<f64>` with **Haar-random unit vectors**
/// (uniformly distributed on `S^{D-1}`).
///
/// Implementation:
/// 1) Draw i.i.d. `Normal(0,1)` for each entry (independent across `k` and `i`).
/// 2) **Normalize** each vector in place to unit L2-norm.
///
/// Parallelism:
/// - Step (1) reuses the parallel filler in `tensor_rand`.
/// - Step (2) uses your `VectorList`’s `normalize` routine.
#[inline]
pub fn fill_haar_vectors(vl: &mut VectorList<f64>) {
    // 1) i.i.d. Gaussian entries
    fill_random(&mut vl.data, RandType::Normal { mean: 0.0, std: 1.0 });

    // 2) in-place L2 normalization per vector
    vl.normalize();
}

/// Allocate and return a `VectorList<f64>` of shape `[dim, n]` filled with
/// Haar-random unit vectors by delegating to `fill_haar_vectors`.
#[inline]
pub fn gen_haar_vectors(dim: usize, n: usize) -> VectorList<f64> {
    let mut vl = VectorList::<f64>::new(dim, n);
    fill_haar_vectors(&mut vl);
    vl
}


// ============================================================================
// ----------------- Decoding Random Nearest-Neighbor vectors -----------------
// ============================================================================

/// Generate one-hot ±1 **nearest-neighbor direction vectors** from displacement codes.
///
/// - `dim`: spatial dimensionality `d`
/// - `n`: number of vectors
/// - `displacements`: length-`n` codes, each a `f64` that is an integer in `[0, 2d)`.
///
/// Encoding rule:
/// - `axis = code / 2`
/// - `sign = if code % 2 == 0 { +1 } else { -1 }`
///
/// The returned `VectorList<isize>` has shape `[dim, n]`, initialized to zeros.
/// For each column `i`, we set `v_i[axis] = sign`.
#[inline]
pub fn gen_neighborhood_vectors(
    dim: usize,
    n: usize,
    displacements: Vec<f64>,
) -> VectorList<isize> {
    assert!(dim > 0, "gen_neighborhood_vectors: dim must be > 0");
    assert!(
        displacements.len() == n,
        "gen_neighborhood_vectors: displacements.len() = {}, expected n = {}",
        displacements.len(),
        n
    );

    let mut vl = VectorList::<isize>::new(dim, n);

    for (i, code_f) in displacements.into_iter().enumerate() {
        // Validate `code_f`
        assert!(
            code_f.is_finite(),
            "gen_neighborhood_vectors: displacement must be finite, got {code_f}"
        );
        assert!(
            code_f >= 0.0,
            "gen_neighborhood_vectors: displacement {code_f} must be >= 0"
        );
        assert!(
            code_f.fract() == 0.0,
            "gen_neighborhood_vectors: displacement {} is not an integer",
            code_f
        );

        let code = code_f as usize;
        assert!(
            code < 2 * dim,
            "gen_neighborhood_vectors: code {} out of range [0, {}) for dim={}",
            code,
            2 * dim,
            dim
        );

        let axis = code / 2;                 // 0..dim-1
        let sign: isize = if code % 2 == 0 { 1 } else { -1 };

        // VectorList::set expects (i: isize, k: isize, val: T) and supports negatives.
        vl.set(i as isize, axis as isize, sign);
    }

    vl
}




// ============================================================================
// ------------------------------- Permutations --------------------------------
// ============================================================================

/// Apply a **single random permutation** of the vector indices `0..n` to the
/// entire `VectorList<T>` in place.
///
/// Semantics:
/// - Treat each vector as an atomic unit (index `i`).
/// - Draw a permutation `π` of `0..n`.
/// - For every dimension `k`, write `row_dst[j] = row_src[π[j]]`.
///
/// Safety & performance:
/// - We first **clone** the underlying data (source snapshot) to avoid aliasing
///   while writing into `vl.data.data`.
/// - We then parallelize over rows (`D` slices of length `n`) using
///   `par_chunks_mut(n)`. Each worker writes to a distinct slice; no races.
/// - Complexity `O(D·n)`; cloning costs `O(D·n)` extra memory.
#[inline]
pub fn permute_vector_list<T: Scalar>(vl: &mut VectorList<T>) {
    let n = vl.num_vectors();
    if n <= 1 {
        return;
    }

    // 1) Build a random permutation π of 0..n
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng_local = rng(); // thread-local RNG; no shared state
    indices.shuffle(&mut rng_local);

    // 2) Snapshot the current data (SoA: [D, n]) to avoid aliasing on write
    let src = vl.data.data.clone();

    // 3) Parallel over rows (each row is a contiguous slice of length n)
    vl.data
        .data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(k, row_dst)| {
            let row_src = &src[k * n..(k + 1) * n];

            // Apply the same permutation to this row so vectors remain intact
            for (j, &i_src) in indices.iter().enumerate() {
                row_dst[j] = row_src[i_src];
            }
        });
}

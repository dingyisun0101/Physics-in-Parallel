// ===================================================================
// ----------------------------- Usage Demo --------------------------
// ===================================================================

/*
Run as a doc-test after adapting the paths to your crate:
*/

use physics_in_parallel::math_fundations::{
    tensor::dense::Tensor as TensorDense, 
    tensor::sparse::Tensor as TensorSparse
};

#[test]
fn sparse_basics_demo() {
    // Build a 2×3 sparse tensor with two nonzeros.
    let s = Sparse::<f64>::from_triplets(
        vec![2, 3],
        vec![(vec![0, 1], 2.0), (vec![1, 2], 3.0)],
    );
    assert_eq!(s.rank(), 2);
    assert_eq!(s.shape(), &[2, 3]);
    assert_eq!(s.nnz(), 2);
    assert_eq!(s.get(&[0, 1]), 2.0);
    assert_eq!(s.get(&[0, 2]), 0.0); // implicit zero

    // Set and prune:
    let mut s2 = s.clone();
    s2.set(&[0, 1], 0.0);          // remove that entry
    s2.add_assign_at(&[0, 1], 5.0); // now becomes 5
    assert_eq!(s2.get(&[0, 1]), 5.0);

    // Elementwise ops preserve sparsity:
    let sum = s + s2.clone();
    assert_eq!(sum.shape(), &[2, 3]);
    assert!(sum.nnz() >= 1);

    // Scalar ops with par_bridge:
    let scaled = s2 * 2.0;
    assert_eq!(scaled.get(&[0, 1]), 10.0);

    // Dense interop:
    let d: Dense<f64> = scaled.to_dense();
    let back = Sparse::from_dense(&d);
    assert_eq!(back.nnz(), scaled.nnz());

    // Casting (f64 -> f32):
    let casted = back.cast_to::<f32>();
    assert_eq!(casted.shape(), &[2, 3]);
}
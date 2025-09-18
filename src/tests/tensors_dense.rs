// tests/tensor_ops.rs

use physics_in_parallel::math_fundations::{
    tensor::dense::Tensor as TensorDense, 
    tensor::sparse::Tensor as TensorSparse
};

//===================================================================
// --------------------------- Usage Example ------------------------
//===================================================================

/*
Run these as doc-tests (`cargo test`) after adapting the `Scalar` trait and
`sparse` module paths to your codebase.
*/

#[test]
fn basic_dense_tensor_demo() {
    // Construct a 2×3 f64 tensor filled with zeros.
    let mut a: Tensor<f64> = Tensor::new(vec![2, 3]);
    assert_eq!(a.data.len(), 6);

    // Safe set/get by multi-index:
    a.set(&[0, 0], 1.0);
    a.set(&[0, 1], 2.0);
    a.set(&[0, 2], 3.0);
    a.set(&[1, 0], 4.0);
    a.set(&[1, 1], 5.0);
    a.set(&[1, 2], 6.0);
    assert_eq!(*a.get(&[1, 2]), 6.0);

    // Parallel in-place map: x ↦ 2x
    a.par_map_inplace(|x| 2.0 * x);
    assert_eq!(*a.get(&[0, 2]), 6.0);

    // Elementwise arithmetic (parallel under the hood):
    let b = a.clone();
    let c = a + b; // add elementwise
    assert_eq!(c.shape, vec![2, 3]);
    assert_eq!(*c.get(&[1, 1]), 20.0); // (5*2) + (5*2) = 20

    // Linear index access (fast path):
    let idx = c.index(&[1, 2]);
    assert_eq!(*c.get_by_idx(idx), 24.0);

    // Casting to another scalar (here it's identity f64→f64):
    let d = c.cast_to::<f64>();
    assert_eq!(d.data, c.data);

    // Optional: sparse roundtrip if you have a compatible sparse module:
    let sp = d.to_sparse();
    let e = Tensor::<f64>::from_sparse(&sp);
    assert_eq!(e.data, d.data);
}

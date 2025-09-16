// tests/tensor_ops.rs

#![allow(clippy::approx_constant)]

use physics_in_parallel::math_fundations::{tensor::dense::Tensor as TensorDense, tensor::sparse::Tensor as TensorSparse};

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps.max(eps * b.abs())
}

#[test]
fn test_basic_division_elementwise() {
    // Arrange
    let a1_str = "[[1.1;2.3;3;4];[5;6;7;8]]";
    let a2_str = "[[-3;2;3;0.4];[5;11;7;8]]";

    let a1: TensorDense<f64> = TensorDense::from_string(a1_str);
    let a2: TensorDense<f64> = TensorDense::from_string(a2_str);

    // Act
    let a3 = a1 / a2;

    // Assert: shape
    assert_eq!(a3.shape(), &[2, 4]);

    // Assert: a few representative values (elementwise division)
    // Expected:
    // [[-0.36666667, 1.15, 1.0, 10.0],
    //  [ 1.0,        0.54545455, 1.0, 1.0]]
    let eps = 1e-9;
    assert!(approx_eq(a3.get(&[0, 0]), -0.366_666_666_666_666_7, eps));
    assert!(approx_eq(a3.get(&[0, 1]),  1.15,                    eps));
    assert!(approx_eq(a3.get(&[0, 2]),  1.0,                     eps));
    assert!(approx_eq(a3.get(&[0, 3]), 10.0,                     eps));
    assert!(approx_eq(a3.get(&[1, 0]),  1.0,                     eps));
    assert!(approx_eq(a3.get(&[1, 1]),  0.545_454_545_454_545_4, eps));
    assert!(approx_eq(a3.get(&[1, 2]),  1.0,                     eps));
    assert!(approx_eq(a3.get(&[1, 3]),  1.0,                     eps));
}

#[test]
fn test_dense_to_sparse_nnz() {
    // Arrange: 3×4 dense tensor with no zeros
    let a1_str = "[[1.1;2.3;3;4];[5;6;7;8];[3;6;7;9]]";
    let a1: TensorDense<f64> = TensorDense::from_string(a1_str);

    // Act
    let b1: TensorSparse<f64> = TensorSparse::from_dense(a1);

    // Assert
    assert_eq!(b1.nnz(), 12);
}

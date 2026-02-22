use ndarray::{ArrayD, IxDyn};

use physics_in_parallel::math::{
    tensor::core::{Dense, Sparse, Tensor},
};

#[test]
fn unified_dense_basic_ops() {
    let mut t = Tensor::<f64, Dense>::empty(&[2, 3]);
    t.set(&[0, 0], 1.0);
    t.set(&[1, 2], 5.0);

    assert_eq!(t.shape(), [2, 3]);
    assert_eq!(t.get(&[0, 0]), 1.0);
    assert_eq!(t.get(&[1, 2]), 5.0);
    assert_eq!(t.get_sum(), 6.0);
}

#[test]
fn unified_sparse_from_dense_and_back() {
    let mut d = Tensor::<f64, Dense>::empty(&[2, 2]);
    d.set(&[0, 0], 2.0);
    d.set(&[1, 1], 3.0);

    let s = d.to_sparse();
    assert_eq!(s.nnz(), 2);
    assert_eq!(s.len_dense(), 4);
    assert_eq!(s.get(&[0, 0]), 2.0);
    assert_eq!(s.get(&[0, 1]), 0.0);

    let d2 = s.to_dense();
    assert_eq!(d2.get(&[1, 1]), 3.0);
}

#[test]
fn unified_dense_ndarray_roundtrip() {
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let t = Tensor::<f64, Dense>::from_ndarray(&a);
    assert_eq!(t.to_ndarray(), a);
}

#[test]
fn unified_sparse_ndarray_roundtrip() {
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 2.0, 0.0, 4.0]).unwrap();
    let t = Tensor::<f64, Sparse>::from_ndarray(&a);
    assert_eq!(t.to_ndarray(), a);
}

#[test]
fn unified_cast_preserves_backend() {
    let mut d = Tensor::<f64, Dense>::empty(&[2]);
    d.set(&[0], 1.25);
    d.set(&[1], -2.5);

    let di = d.cast_to::<i64>();
    assert_eq!(di.get(&[0]), 1);
    assert_eq!(di.get(&[1]), -2);

    let s = d.to_sparse();
    let si = s.cast_to::<i64>();
    assert_eq!(si.get(&[0]), 1);
    assert_eq!(si.get(&[1]), -2);
}

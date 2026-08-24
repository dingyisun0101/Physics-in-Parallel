use physics_in_parallel::prelude::advanced::{Dense, Sparse};
use physics_in_parallel::prelude::basic::{Complex, ScalarCastError, Tensor};

fn assert_tensor_values(tensor: &Tensor<f64, Dense>, shape: &[usize], expected: &[f64]) {
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.as_slice(), expected);
}

fn dense_2x3() -> Tensor<f64, Dense> {
    let mut t = Tensor::<f64, Dense>::empty(&[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            t.set(&[i, j], (1 + i * 3 + j) as f64);
        }
    }
    t
}

#[test]
fn dense_tensor_basic_metadata_access_and_wrapping() {
    let t = dense_2x3();
    let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.rank(), 2);
    assert_eq!(t.size(), 6);
    assert_eq!(t.sum(), expected.iter().sum::<f64>());
    assert_tensor_values(&t, &[2, 3], &expected);

    assert_eq!(t.get(&[-1, -1]), expected[5]);
    assert_eq!(t.get(&[2, 3]), expected[0]);
}

#[test]
fn dense_tensor_type_conversion_uses_scalar_cast_contract() {
    let mut t = Tensor::<f64, Dense>::empty(&[2]);
    t.set(&[0], 1.25);
    t.set(&[1], 2.5);

    let casted = t.try_cast_to::<f32>().unwrap();
    assert_eq!(casted.get(&[0]), 1.25f32);
    assert_eq!(casted.get(&[1]), 2.5f32);

    t.set(&[1], f64::MAX);
    let err = t.try_cast_to::<f32>().unwrap_err();
    assert!(matches!(err, ScalarCastError::RealPartOutOfRange { .. }));

    let mut z = Tensor::<Complex<f64>, Dense>::empty(&[1]);
    z.set(&[0], Complex::new(3.0, 4.0));
    let real = z.cast_to::<f64>();
    assert_eq!(real.get(&[0]), 3.0);
}

#[test]
fn dense_tensor_elementwise_ops_match_manual_values() {
    let a = dense_2x3();
    let expected_b = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

    let scaled = a.scalar_mul(2.0);
    assert_tensor_values(&scaled, &[2, 3], &expected_b);

    let shifted = a.map(|x| x + 1.0);
    assert_tensor_values(&shifted, &[2, 3], &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    let elem_mul = a.elem_mul(&scaled);
    assert_tensor_values(&elem_mul, &[2, 3], &[2.0, 8.0, 18.0, 32.0, 50.0, 72.0]);

    let elem_div = scaled.elem_div(&a);
    assert_tensor_values(&elem_div, &[2, 3], &[2.0; 6]);
}

#[test]
fn sparse_tensor_basic_elementwise_semantics_match_stored_entries() {
    let mut a = Tensor::<i64, Sparse>::empty(&[3, 3]);
    a.set(&[0, 0], 2);
    a.set(&[1, 1], -3);
    a.set(&[2, 2], 0);

    assert_eq!(a.rank(), 2);
    assert_eq!(a.size(), 9);
    assert_eq!(a.nnz(), 2);
    assert_eq!(a.sum(), -1);
    assert_eq!(a.get(&[2, 2]), 0);

    let abs = a.abs();
    assert_eq!(abs.get(&[0, 0]), 2);
    assert_eq!(abs.get(&[1, 1]), 3);
    assert_eq!(abs.nnz(), 2);

    let squared = a.norm_sqr();
    assert_eq!(squared.get(&[0, 0]), 4);
    assert_eq!(squared.get(&[1, 1]), 9);
    assert_eq!(squared.nnz(), 2);
}

#[test]
fn dense_and_sparse_facades_interoperate_for_common_math_ops() {
    let dense = Tensor::<i64, Dense>::from_vec(&[3], vec![2, 3, 4]);
    let sparse = Tensor::<i64, Sparse>::from_triplets(
        vec![3],
        vec![(vec![0], 10), (vec![2], -1), (vec![1], 0)],
    );

    assert_eq!(dense.dot(&sparse), 16);
    assert_eq!(sparse.dot(&dense), 16);

    let dense_product = dense.elem_mul(&sparse);
    assert_eq!(dense_product.get(&[0]), 20);
    assert_eq!(dense_product.get(&[1]), 0);
    assert_eq!(dense_product.get(&[2]), -4);

    let sparse_product = sparse.elem_mul(&dense);
    assert_eq!(sparse_product.get(&[0]), 20);
    assert_eq!(sparse_product.get(&[1]), 0);
    assert_eq!(sparse_product.get(&[2]), -4);
    assert_eq!(sparse_product.nnz(), 2);
}

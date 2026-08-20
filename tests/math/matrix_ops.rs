use physics_in_parallel::prelude::advanced::{
    DiagonalMatrix, SparseMatrix, StrictUpperTriangularMatrix, SymmetricMatrix,
    UpperTriangularMatrix,
};
use physics_in_parallel::prelude::basic::{Complex, DenseMatrix, MatrixError};

use crate::matrix_support::{assert_panics, print_matrix};

#[test]
fn type_preserving_dense_matrix_ops_match_manual_results() {
    let a = DenseMatrix::<i64>::from_vec(2, 2, vec![1, 2, 3, 4]);
    let b = DenseMatrix::<i64>::from_vec(2, 2, vec![10, 20, 30, 40]);

    let sum = a.add(&b);
    assert_eq!(sum.get(0, 0), 11);
    assert_eq!(sum.get(1, 1), 44);
    print_matrix("dense sum", &sum);

    let diff = b.sub(&a);
    assert_eq!(diff.get(0, 1), 18);
    assert_eq!(diff.get(1, 0), 27);

    let elem_mul = a.elem_mul(&b);
    assert_eq!(elem_mul.get(0, 1), 40);
    assert_eq!(elem_mul.get(1, 1), 160);

    let elem_div = b.elem_div(&a);
    assert_eq!(elem_div.get(0, 0), 10);
    assert_eq!(elem_div.get(1, 1), 10);

    let scaled = a.scalar_mul(3);
    assert_eq!(scaled.get(0, 1), 6);
    assert_eq!(scaled.get(1, 0), 9);

    let signed = DenseMatrix::<i64>::from_vec(2, 2, vec![-1, 2, -3, 4]);
    let absolute = signed.abs();
    assert_eq!(absolute.get(0, 0), 1);
    assert_eq!(absolute.get(0, 1), 2);
    assert_eq!(absolute.get(1, 0), 3);
    assert_eq!(absolute.get(1, 1), 4);

    assert_eq!(a.max_abs_real(), 4);
}

#[test]
fn batched_matrix_application_infers_vectors_and_reuses_output() {
    let matrix = DenseMatrix::<i64>::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
    let input = [1, 0, 1, 2, 1, 0];
    let mut output = [0; 4];
    matrix.mul_vectors_into(&input, &mut output).unwrap();
    assert_eq!(output, [4, 10, 4, 13]);

    assert_eq!(matrix.get_flat(-1), 6);
    let mut writable = matrix.clone();
    writable.set_flat(6, 9);
    assert_eq!(writable.get(0, 0), 9);

    assert!(matches!(
        matrix.mul_vectors_into(&[1, 2], &mut []),
        Err(MatrixError::BatchInputLength {
            columns: 3,
            actual: 2
        })
    ));
}

#[test]
fn maximum_absolute_entry_uses_complete_matrix_semantics() {
    let dense = DenseMatrix::<f64>::from_vec(2, 2, vec![-1.0, 7.5, -3.0, 2.0]);
    assert_eq!(dense.max_abs_real(), 7.5);

    let sparse = SparseMatrix::<f64>::from_triplets(3, 3, vec![(0, 2, -9.0), (2, 1, 4.0)]);
    assert_eq!(sparse.max_abs_real(), 9.0);

    let mut symmetric = SymmetricMatrix::<f64>::empty(3, 3);
    symmetric.set(0, 2, -11.0);
    assert_eq!(symmetric.max_abs_real(), 11.0);

    let with_nan = DenseMatrix::<f64>::from_vec(1, 2, vec![1.0, f64::NAN]);
    assert!(with_nan.max_abs_real().is_nan());
}

#[test]
fn type_preserving_sparse_matrix_ops_keep_sparse_backend() {
    let sparse = SparseMatrix::<i64>::from_triplets(2, 2, vec![(0, 0, 10), (1, 1, -1)]);
    let dense = DenseMatrix::<i64>::from_vec(2, 2, vec![1, 2, 3, 4]);

    let sum = sparse.add(&dense);
    assert_eq!(sum.nnz(), 4);
    assert_eq!(sum.get(0, 0), 11);
    assert_eq!(sum.get(0, 1), 2);
    assert_eq!(sum.get(1, 0), 3);
    assert_eq!(sum.get(1, 1), 3);
    print_matrix("sparse-preserving sum", &sum);

    let product = sparse.matmul(&dense);
    assert_eq!(product.nnz(), 4);
    assert_eq!(product.get(0, 0), 10);
    assert_eq!(product.get(0, 1), 20);
    assert_eq!(product.get(1, 0), -3);
    assert_eq!(product.get(1, 1), -4);
    print_matrix("sparse-preserving matrix product", &product);
}

#[test]
fn transpose_hermitian_transpose_trace_and_matmul_match_manual_results() {
    let matrix = DenseMatrix::<i64>::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
    let transposed = matrix.transpose();
    assert_eq!(transposed.shape(), [3, 2]);
    assert_eq!(transposed.get(0, 1), 4);
    assert_eq!(transposed.get(2, 0), 3);
    print_matrix("dense transpose", &transposed);

    let left = DenseMatrix::<i64>::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
    let right = DenseMatrix::<i64>::from_vec(3, 2, vec![7, 8, 9, 10, 11, 12]);
    let product = left.matmul(&right);
    assert_eq!(product.shape(), [2, 2]);
    assert_eq!(product.get(0, 0), 58);
    assert_eq!(product.get(0, 1), 64);
    assert_eq!(product.get(1, 0), 139);
    assert_eq!(product.get(1, 1), 154);
    assert_eq!(product.trace(), 212);
    print_matrix("dense matrix product", &product);

    let complex = DenseMatrix::<Complex<f64>>::from_vec(
        2,
        2,
        vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -4.0),
            Complex::new(-5.0, 6.0),
            Complex::new(7.0, 8.0),
        ],
    );
    let hermitian = complex.hermitian_transpose();
    assert_eq!(hermitian.get(0, 0), Complex::new(1.0, -2.0));
    assert_eq!(hermitian.get(1, 0), Complex::new(3.0, 4.0));
    assert_eq!(hermitian.get(0, 1), Complex::new(-5.0, -6.0));
}

#[test]
fn explicit_to_dense_ops_cover_backend_changing_matrix_results() {
    let mut upper = UpperTriangularMatrix::<i64>::empty(3, 3);
    upper.set(0, 2, 9);
    upper.set(1, 1, 5);

    assert_panics(|| {
        let _ = upper.transpose();
    });

    let transposed = upper.transpose_to_dense();
    assert_eq!(transposed.get(2, 0), 9);
    assert_eq!(transposed.get(1, 1), 5);
    assert_eq!(transposed.get(0, 2), 0);
    print_matrix("upper transpose converted to dense", &transposed);

    let dense = DenseMatrix::<i64>::from_vec(3, 3, vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
    let product = upper.matmul_to_dense(&dense);
    assert_eq!(product.get(0, 2), 27);
    assert_eq!(product.get(1, 1), 10);
    print_matrix("upper times dense converted to dense", &product);

    let scaled = upper.scalar_mul_to_dense(2);
    assert_eq!(scaled.get(0, 2), 18);
    assert_eq!(scaled.get(1, 1), 10);
}

#[test]
fn structured_preserving_ops_succeed_only_when_result_fits_left_backend() {
    let mut diagonal = DiagonalMatrix::<i64>::empty(3, 3);
    diagonal.set(0, 0, 2);
    diagonal.set(1, 1, -1);
    diagonal.set(2, 2, 4);

    let mut symmetric = SymmetricMatrix::<i64>::empty(3, 3);
    symmetric.set(0, 1, 7);

    let diagonal_sum = diagonal.add(&diagonal);
    assert_eq!(diagonal_sum.get(0, 0), 4);
    assert_eq!(diagonal_sum.get(0, 1), 0);

    assert_panics(|| {
        let _ = diagonal.add(&symmetric);
    });

    let dense_sum = diagonal.add_to_dense(&symmetric);
    assert_eq!(dense_sum.get(0, 0), 2);
    assert_eq!(dense_sum.get(0, 1), 7);
    assert_eq!(dense_sum.get(1, 0), 7);
    print_matrix("diagonal plus symmetric converted to dense", &dense_sum);

    let mut strict_upper = StrictUpperTriangularMatrix::<i64>::empty(3, 3);
    strict_upper.set(0, 1, 2);
    strict_upper.set(1, 2, 3);
    let strict_scaled = strict_upper.scalar_mul(4);
    assert_eq!(strict_scaled.get(0, 1), 8);
    assert_eq!(strict_scaled.get(1, 2), 12);
    assert_eq!(strict_scaled.get(1, 1), 0);
}

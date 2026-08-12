use physics_in_parallel::math::tensor::{DenseMatrix, MatrixError, SparseMatrix};

use crate::matrix_support::print_matrix;

#[test]
fn dense_matrix_construction_access_fill_print_and_cast_are_consistent() {
    let mut dense = DenseMatrix::<i64>::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);

    assert_eq!(dense.rows(), 2);
    assert_eq!(dense.cols(), 3);
    assert_eq!(dense.shape(), [2, 3]);
    assert_eq!(dense.size(), 6);
    assert_eq!(dense.get(0, 0), 1);
    assert_eq!(dense.get(1, 2), 6);
    assert_eq!(dense.get(-1, -1), 6);

    dense.set(-1, -1, 9);
    assert_eq!(dense.get(1, 2), 9);
    print_matrix("dense after wrapped set", &dense);

    dense.fill(7);
    assert_eq!(dense.get(0, 0), 7);
    assert_eq!(dense.get(1, 2), 7);
    print_matrix("dense after fill", &dense);

    let generated = DenseMatrix::<i64>::from_fn(2, 2, |row, col| (10 * row + col) as i64);
    assert_eq!(generated.get(1, 0), 10);
    print_matrix("dense from_fn", &generated);

    let casted = generated.cast_to::<f64>();
    assert_eq!(casted.get(1, 0), 10.0);
}

#[test]
fn fallible_dense_construction_reports_shape_and_storage_errors() {
    assert!(matches!(
        DenseMatrix::<f64>::try_from_vec(0, 2, Vec::new()),
        Err(MatrixError::InvalidShape { rows: 0, cols: 2 })
    ));
    assert!(matches!(
        DenseMatrix::<f64>::try_from_vec(2, 2, vec![1.0, 2.0, 3.0]),
        Err(MatrixError::DataLengthMismatch {
            expected: 4,
            actual: 3,
        })
    ));
}

#[test]
fn matrix_vector_application_reuses_caller_output_for_dense_and_sparse_storage() {
    let dense = DenseMatrix::<f64>::try_from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid dense matrix");
    let mut output = [0.0; 2];
    dense
        .mul_vector_into(&[2.0, 1.0, -1.0], &mut output)
        .expect("matching vector dimensions");
    assert_eq!(output, [1.0, 7.0]);

    let sparse = dense.to_sparse();
    sparse
        .mul_vector_into(&[2.0, 1.0, -1.0], &mut output)
        .expect("matching vector dimensions");
    assert_eq!(output, [1.0, 7.0]);

    assert_eq!(
        dense.mul_vector_into(&[1.0], &mut output),
        Err(MatrixError::InputLength {
            expected: 3,
            actual: 1,
        })
    );
    assert_eq!(
        dense.mul_vector_into(&[1.0, 2.0, 3.0], &mut [0.0]),
        Err(MatrixError::OutputLength {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn dense_and_sparse_rank_n_backends_convert_without_changing_logical_values() {
    let dense = DenseMatrix::<i64>::from_vec(2, 3, vec![2, 0, 0, 0, 0, 5]);
    let sparse = dense.to_sparse();

    assert_eq!(sparse.shape(), [2, 3]);
    assert_eq!(sparse.nnz(), 2);
    assert_eq!(sparse.get(0, 0), 2);
    assert_eq!(sparse.get(1, 2), 5);
    assert_eq!(sparse.get(0, 1), 0);
    print_matrix("sparse converted from dense", &sparse);

    let dense_again = sparse.to_dense();
    assert_eq!(dense_again.get(0, 0), 2);
    assert_eq!(dense_again.get(1, 2), 5);
    assert_eq!(dense_again.get(0, 1), 0);
    print_matrix("dense converted back from sparse", &dense_again);
}

#[test]
fn sparse_matrix_triplets_wrapping_zero_pruning_print_and_cast_are_consistent() {
    let mut sparse =
        SparseMatrix::<i64>::from_triplets(3, 3, vec![(0, 0, 4), (1, 2, -6), (2, 2, 0)]);

    assert_eq!(sparse.nnz(), 2);
    assert_eq!(sparse.get(0, 0), 4);
    assert_eq!(sparse.get(1, 2), -6);
    assert_eq!(sparse.get(2, 2), 0);

    sparse.set(-1, -1, 8);
    assert_eq!(sparse.get(2, 2), 8);
    assert_eq!(sparse.nnz(), 3);

    sparse.set(0, 0, 0);
    assert_eq!(sparse.get(0, 0), 0);
    assert_eq!(sparse.nnz(), 2);
    print_matrix("sparse after wrapped set and zero pruning", &sparse);

    let casted = sparse
        .try_cast_to::<f64>()
        .expect("i64 to f64 cast should succeed");
    assert_eq!(casted.get(1, 2), -6.0);
}

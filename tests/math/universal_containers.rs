use physics_in_parallel::math::{Backend, Matrix, Tensor, TensorError, VectorList};

#[test]
fn reusable_arithmetic_covers_every_backend_combination() {
    for left in [Backend::Dense, Backend::Sparse] {
        for right in [Backend::Dense, Backend::Sparse] {
            for destination in [Backend::Dense, Backend::Sparse] {
                let a = Tensor::from_values(&[3], left, vec![0.0_f64, 2.0, -3.0]).unwrap();
                let b = Tensor::from_values(&[3], right, vec![0.0, -2.0, 4.0]).unwrap();
                let mut out = Tensor::filled(&[3], destination, 99.0).unwrap();
                a.add_into(&b, &mut out).unwrap();
                assert_eq!(out.values().collect::<Vec<_>>(), [0.0, 0.0, 1.0]);
                a.multiply_into(&b, &mut out).unwrap();
                assert_eq!(out.values().collect::<Vec<_>>(), [0.0, -4.0, -12.0]);
                a.divide_into(&b, &mut out).unwrap();
                assert!(out.get(&[0]).unwrap().is_nan());
                assert_eq!(out.get(&[2]).unwrap(), -0.75);
                assert_eq!(out.backend(), destination);
                let before = out.values().collect::<Vec<_>>();
                let bad = Tensor::zeros(&[1], left).unwrap();
                assert!(a.add_into(&bad, &mut out).is_err());
                for (a, b) in before.iter().zip(out.values()) {
                    assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan());
                }
            }
        }
    }
}

#[test]
fn sparse_kernels_handle_huge_logical_shapes_and_nonfinite_zeros() {
    let size = 1_000_000_000;
    let a = Tensor::from_entries(&[size], Backend::Sparse, [(vec![7], 3.0)]).unwrap();
    let b = a.scale(2.0);
    assert_eq!(b.sum(), 6.0);
    assert_eq!(a.add(&b).unwrap().sum(), 9.0);
    assert_eq!(b.cast::<f32>().unwrap().get(&[7]).unwrap(), 6.0);
    let matrix = Tensor::from_entries(&[size, 2], Backend::Sparse, [(vec![7, 1], 3.0)]).unwrap();
    assert_eq!(matrix.transpose().unwrap().get(&[1, 7]).unwrap(), 3.0);
    let mut cleared = b;
    cleared.fill(0.0);
    assert_eq!(cleared.sum(), 0.0);
    let a = Tensor::<f64>::zeros(&[3], Backend::Sparse).unwrap();
    assert!(a.scale(f64::INFINITY).values().all(f64::is_nan));
}

#[test]
fn matrix_output_and_vector_copies_validate_before_mutating() {
    let a = Matrix::from_values(2, 3, Backend::Dense, vec![1_i64, 2, 3, 4, 5, 6]).unwrap();
    let b = Matrix::from_values(3, 2, Backend::Dense, vec![7_i64, 8, 9, 10, 11, 12]).unwrap();
    for backend in [Backend::Dense, Backend::Sparse] {
        let mut out = Matrix::zeros(2, 2, backend).unwrap();
        a.matmul_into(&b, &mut out).unwrap();
        assert_eq!(out.values().collect::<Vec<_>>(), [58, 64, 139, 154]);
        let vectors = VectorList::from_values(2, 2, backend, vec![1_i64, 2, 3, 4]).unwrap();
        let mut row = [99; 2];
        vectors.vector_into(1, &mut row).unwrap();
        assert_eq!(row, [3, 4]);
        vectors.axis_into(1, &mut row).unwrap();
        assert_eq!(row, [2, 4]);
        assert!(vectors.axis_into(2, &mut row).is_err());
        assert_eq!(row, [2, 4]);
    }
}

#[test]
fn derived_shapes_are_rejected_without_materializing_sparse_inputs() {
    let extent = 1_usize << (usize::BITS / 2);
    let lhs = Tensor::<f64>::zeros(&[extent, 1], Backend::Sparse).unwrap();
    let rhs = Tensor::<f64>::zeros(&[1, extent], Backend::Sparse).unwrap();
    assert!(matches!(
        lhs.matmul(&rhs),
        Err(TensorError::ShapeProductOverflow { .. })
    ));
    let vector = Tensor::<f64>::zeros(&[extent], Backend::Sparse).unwrap();
    assert!(matches!(
        vector.wedge(&vector),
        Err(TensorError::ShapeProductOverflow { .. })
    ));
    let lhs = Tensor::<f64>::zeros(&[isize::MAX as usize, 1], Backend::Sparse).unwrap();
    let rhs = Tensor::<f64>::zeros(&[1, 2], Backend::Sparse).unwrap();
    assert!(matches!(
        lhs.matmul(&rhs),
        Err(TensorError::IndexSpaceOverflow { .. })
    ));
}

#[test]
fn tensor_construction_selects_backend_without_changing_the_public_type() {
    let values = vec![1_i64, 0, 0, 2];
    let dense = Tensor::from_values(&[2, 2], Backend::Dense, values.clone()).unwrap();
    let sparse = Tensor::from_values(&[2, 2], Backend::Sparse, values).unwrap();

    assert_eq!(dense.backend(), Backend::Dense);
    assert_eq!(sparse.backend(), Backend::Sparse);
    assert_eq!(
        dense.values().collect::<Vec<_>>(),
        sparse.values().collect::<Vec<_>>()
    );
}

#[test]
fn coordinate_access_is_strict_and_fallible() {
    let mut tensor = Tensor::<i64>::zeros(&[2, 3], Backend::Sparse).unwrap();
    tensor.set(&[1, 2], 7).unwrap();
    assert_eq!(tensor.get(&[1, 2]).unwrap(), 7);
    assert!(matches!(
        tensor.get(&[1]),
        Err(TensorError::RankMismatch { .. })
    ));
    assert!(matches!(
        tensor.get(&[2, 0]),
        Err(TensorError::CoordinateOutOfBounds { axis: 0, .. })
    ));
}

#[test]
fn sparse_construction_rejects_duplicate_coordinates() {
    let error = Tensor::from_entries(
        &[2, 2],
        Backend::Sparse,
        [(vec![0, 1], 3_i64), (vec![0, 1], 4_i64)],
    )
    .unwrap_err();
    assert!(matches!(error, TensorError::DuplicateCoordinate { .. }));
}

#[test]
fn empty_builders_enforce_backend_specific_initialization() {
    let mut dense = Tensor::<i64>::empty(&[2], Backend::Dense).unwrap();
    dense.set(&[0], 7).unwrap();
    assert_eq!(dense.backend(), Backend::Dense);
    assert!(matches!(
        dense.finish(),
        Err(TensorError::IncompleteInitialization { remaining: 1 })
    ));

    let mut dense = Tensor::<i64>::empty(&[2], Backend::Dense).unwrap();
    dense.set(&[0], 7).unwrap();
    dense.set(&[1], 0).unwrap();
    let dense = dense.finish().unwrap();
    assert_eq!(dense.values().collect::<Vec<_>>(), vec![7, 0]);

    let mut sparse = Tensor::<i64>::empty(&[2], Backend::Sparse).unwrap();
    sparse.set(&[0], 7).unwrap();
    let sparse = sparse.finish().unwrap();
    assert_eq!(sparse.values().collect::<Vec<_>>(), vec![7, 0]);
}

#[test]
fn semantic_container_empty_builders_preserve_the_selected_backend() {
    let mut matrix = Matrix::<i64>::empty(1, 2, Backend::Dense).unwrap();
    matrix.set(0, 0, 3).unwrap();
    matrix.set(0, 1, 4).unwrap();
    let matrix = matrix.finish().unwrap();
    assert_eq!(matrix.backend(), Backend::Dense);
    assert_eq!(matrix.values().collect::<Vec<_>>(), vec![3, 4]);

    let mut vectors = VectorList::<i64>::empty(2, 2, Backend::Sparse).unwrap();
    vectors.set_vector(1, &[5, 6]).unwrap();
    let vectors = vectors.finish().unwrap();
    assert_eq!(vectors.backend(), Backend::Sparse);
    assert_eq!(vectors.values().collect::<Vec<_>>(), vec![0, 0, 5, 6]);
}

#[test]
fn zero_construction_respects_the_selected_backend() {
    let dense = Tensor::<i64>::zeros(&[2], Backend::Dense).unwrap();
    let sparse = Tensor::<i64>::zeros(&[2], Backend::Sparse).unwrap();
    assert_eq!(dense.backend(), Backend::Dense);
    assert_eq!(sparse.backend(), Backend::Sparse);
    assert_eq!(dense.values().collect::<Vec<_>>(), vec![0, 0]);
    assert_eq!(sparse.values().collect::<Vec<_>>(), vec![0, 0]);
}

#[test]
fn allocating_and_into_operations_follow_the_documented_backend_owner() {
    let dense = Tensor::from_values(&[2], Backend::Dense, vec![1_i64, 2]).unwrap();
    let sparse = Tensor::from_entries(&[2], Backend::Sparse, [(vec![1], 3_i64)]).unwrap();

    let dense_result = dense.add(&sparse).unwrap();
    let sparse_result = sparse.add(&dense).unwrap();
    assert_eq!(dense_result.backend(), Backend::Dense);
    assert_eq!(sparse_result.backend(), Backend::Sparse);
    assert_eq!(dense_result.values().collect::<Vec<_>>(), vec![1, 5]);
    assert_eq!(sparse_result.values().collect::<Vec<_>>(), vec![1, 5]);

    let mut output = Tensor::zeros(&[2], Backend::Sparse).unwrap();
    dense.add_into(&sparse, &mut output).unwrap();
    assert_eq!(output.backend(), Backend::Sparse);
    assert_eq!(output.values().collect::<Vec<_>>(), vec![1, 5]);
}

#[test]
fn explicit_conversion_and_serde_preserve_backend() {
    let mut tensor = Tensor::from_values(&[2, 2], Backend::Dense, vec![0_i64, 1, 0, 2]).unwrap();
    tensor.set_backend(Backend::Sparse);
    assert_eq!(tensor.backend(), Backend::Sparse);

    let json = serde_json::to_string(&tensor).unwrap();
    assert!(json.contains("\"backend\":\"sparse\""));
    assert!(!json.contains("\"storage\""));
    let restored: Tensor<i64> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.backend(), Backend::Sparse);
    assert_eq!(restored, tensor);

    tensor.set_backend(Backend::Dense);
    assert_eq!(tensor.backend(), Backend::Dense);
}

#[test]
fn matrix_mixed_backend_math_preserves_receiver() {
    let dense = Matrix::from_values(2, 2, Backend::Dense, vec![1_i64, 2, 3, 4]).unwrap();
    let sparse = Matrix::from_entries(2, 2, Backend::Sparse, [(0, 0, 2_i64), (1, 1, 3)]).unwrap();

    let left_dense = dense.matmul(&sparse).unwrap();
    let left_sparse = sparse.matmul(&dense).unwrap();
    assert_eq!(left_dense.backend(), Backend::Dense);
    assert_eq!(left_sparse.backend(), Backend::Sparse);
    assert_eq!(left_dense.values().collect::<Vec<_>>(), vec![2, 6, 6, 12]);
    assert_eq!(left_sparse.values().collect::<Vec<_>>(), vec![2, 4, 9, 12]);
}

#[test]
fn vector_list_supports_both_backends_through_one_api() {
    let dense =
        VectorList::from_values(2, 2, Backend::Dense, vec![3.0_f64, 4.0, 0.0, 0.0]).unwrap();
    let mut sparse =
        VectorList::from_entries(2, 2, Backend::Sparse, [(0, 0, 3.0_f64), (0, 1, 4.0)]).unwrap();

    assert_eq!(dense.norms(), vec![5.0, 0.0]);
    assert_eq!(sparse.norms(), vec![5.0, 0.0]);
    sparse.normalize().unwrap();
    assert_eq!(sparse.backend(), Backend::Sparse);
    let normalized = sparse.vector(0).unwrap();
    assert!((normalized[0] - 0.6).abs() < 1e-12);
    assert!((normalized[1] - 0.8).abs() < 1e-12);
}

#[test]
fn public_containers_and_errors_are_thread_compatible() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<Tensor<f64>>();
    assert_send_sync::<Matrix<f64>>();
    assert_send_sync::<VectorList<f64>>();
    assert_send_sync::<TensorError>();
}

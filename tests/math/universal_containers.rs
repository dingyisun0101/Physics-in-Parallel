use physics_in_parallel::math::{Matrix, StorageKind, Tensor, TensorError, VectorList};

#[test]
fn tensor_construction_selects_storage_without_changing_the_public_type() {
    let dense = Tensor::from_dense(&[2, 2], vec![1_i64, 0, 0, 2]).unwrap();
    let sparse =
        Tensor::from_sparse_entries(&[2, 2], [(vec![0, 0], 1_i64), (vec![1, 1], 2)]).unwrap();

    assert_eq!(dense.storage_kind(), StorageKind::Dense);
    assert_eq!(sparse.storage_kind(), StorageKind::Sparse);
    assert_eq!(
        dense.values().collect::<Vec<_>>(),
        sparse.values().collect::<Vec<_>>()
    );
}

#[test]
fn coordinate_access_is_strict_and_fallible() {
    let mut tensor = Tensor::<i64>::zeros(&[2, 3]).unwrap();
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
    let error = Tensor::from_sparse_entries(&[2, 2], [(vec![0, 1], 3_i64), (vec![0, 1], 4_i64)])
        .unwrap_err();
    assert!(matches!(error, TensorError::DuplicateCoordinate { .. }));
}

#[test]
fn allocating_and_into_operations_follow_the_documented_storage_owner() {
    let dense = Tensor::from_dense(&[2], vec![1_i64, 2]).unwrap();
    let sparse = Tensor::from_sparse_entries(&[2], [(vec![1], 3_i64)]).unwrap();

    let dense_result = dense.add(&sparse).unwrap();
    let sparse_result = sparse.add(&dense).unwrap();
    assert_eq!(dense_result.storage_kind(), StorageKind::Dense);
    assert_eq!(sparse_result.storage_kind(), StorageKind::Sparse);
    assert_eq!(dense_result.values().collect::<Vec<_>>(), vec![1, 5]);
    assert_eq!(sparse_result.values().collect::<Vec<_>>(), vec![1, 5]);

    let mut output = Tensor::zeros(&[2]).unwrap();
    dense.add_into(&sparse, &mut output).unwrap();
    assert_eq!(output.storage_kind(), StorageKind::Sparse);
    assert_eq!(output.values().collect::<Vec<_>>(), vec![1, 5]);
}

#[test]
fn explicit_conversion_and_serde_preserve_storage() {
    let mut tensor = Tensor::from_dense(&[2, 2], vec![0_i64, 1, 0, 2]).unwrap();
    tensor.make_sparse();
    assert_eq!(tensor.storage_kind(), StorageKind::Sparse);

    let json = serde_json::to_string(&tensor).unwrap();
    let restored: Tensor<i64> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.storage_kind(), StorageKind::Sparse);
    assert_eq!(restored, tensor);

    tensor.make_dense();
    assert_eq!(tensor.storage_kind(), StorageKind::Dense);
}

#[test]
fn matrix_mixed_storage_math_preserves_receiver() {
    let dense = Matrix::from_dense(2, 2, vec![1_i64, 2, 3, 4]).unwrap();
    let sparse = Matrix::from_sparse_entries(2, 2, [(0, 0, 2_i64), (1, 1, 3)]).unwrap();

    let left_dense = dense.matmul(&sparse).unwrap();
    let left_sparse = sparse.matmul(&dense).unwrap();
    assert_eq!(left_dense.storage_kind(), StorageKind::Dense);
    assert_eq!(left_sparse.storage_kind(), StorageKind::Sparse);
    assert_eq!(left_dense.values().collect::<Vec<_>>(), vec![2, 6, 6, 12]);
    assert_eq!(left_sparse.values().collect::<Vec<_>>(), vec![2, 4, 9, 12]);
}

#[test]
fn vector_list_supports_both_storage_kinds_through_one_api() {
    let dense = VectorList::from_dense(2, 2, vec![3.0_f64, 4.0, 0.0, 0.0]).unwrap();
    let mut sparse = VectorList::from_sparse_entries(2, 2, [(0, 0, 3.0_f64), (0, 1, 4.0)]).unwrap();

    assert_eq!(dense.norms(), vec![5.0, 0.0]);
    assert_eq!(sparse.norms(), vec![5.0, 0.0]);
    sparse.normalize().unwrap();
    assert_eq!(sparse.storage_kind(), StorageKind::Sparse);
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

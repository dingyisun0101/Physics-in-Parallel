use physics_in_parallel::math::tensor::core::{dense, sparse, Dense, Sparse, Tensor};
use physics_in_parallel::math::tensor::core::tensor_trait::TensorTrait;
use physics_in_parallel::math::tensor::{Matrix, Tensor2D, VectorList};

#[test]
fn dense_tensor_serde_roundtrip() {
    let mut tensor = dense::Tensor::<f64>::empty(&[2, 2]);
    tensor.set(&[0, 0], 1.0);
    tensor.set(&[1, 1], 4.0);

    let json = serde_json::to_value(&tensor).expect("dense tensor should serialize");
    assert_eq!(json["kind"], "tensor");
    assert_eq!(json["shape"], serde_json::json!([2, 2]));
    assert_eq!(json["data"], serde_json::json!([1.0, 0.0, 0.0, 4.0]));

    let restored: dense::Tensor<f64> =
        serde_json::from_value(json).expect("dense tensor should deserialize");
    assert_eq!(restored.shape, vec![2, 2]);
    assert_eq!(restored.get(&[0, 0]), 1.0);
    assert_eq!(restored.get(&[1, 1]), 4.0);
}

#[test]
fn sparse_tensor_serde_roundtrip() {
    let mut tensor = sparse::Tensor::<f64>::empty(&[2, 3]);
    tensor.set(&[0, 1], 2.0);
    tensor.set(&[1, 2], 5.0);

    let json = serde_json::to_value(&tensor).expect("sparse tensor should serialize");
    assert_eq!(json["kind"], "tensor_sparse");
    assert_eq!(json["shape"], serde_json::json!([2, 3]));
    assert_eq!(json["data"], serde_json::json!([0.0, 2.0, 0.0, 0.0, 0.0, 5.0]));

    let restored: sparse::Tensor<f64> =
        serde_json::from_value(json).expect("sparse tensor should deserialize");
    assert_eq!(restored.shape(), &[2, 3]);
    assert_eq!(restored.get(&[0, 1]), 2.0);
    assert_eq!(restored.get(&[1, 2]), 5.0);
    assert_eq!(restored.nnz(), 2);
}

#[test]
fn unified_tensor_serde_roundtrip() {
    let mut dense_tensor = Tensor::<f64, Dense>::empty(&[3]);
    dense_tensor.set(&[0], 1.5);
    dense_tensor.set(&[2], -2.0);
    let dense_json = serde_json::to_value(&dense_tensor).expect("unified dense should serialize");
    let dense_restored: Tensor<f64, Dense> =
        serde_json::from_value(dense_json).expect("unified dense should deserialize");
    assert_eq!(dense_restored.get(&[0]), 1.5);
    assert_eq!(dense_restored.get(&[2]), -2.0);

    let sparse_tensor = dense_tensor.to_sparse();
    let sparse_json =
        serde_json::to_value(&sparse_tensor).expect("unified sparse should serialize");
    let sparse_restored: Tensor<f64, Sparse> =
        serde_json::from_value(sparse_json).expect("unified sparse should deserialize");
    assert_eq!(sparse_restored.get(&[0]), 1.5);
    assert_eq!(sparse_restored.get(&[2]), -2.0);
}

#[test]
fn tensor2d_matrix_and_vector_list_serde_roundtrip() {
    let mut tensor2d = Tensor2D::<f64>::empty(2, 2);
    tensor2d.set(0, 0, 1.0);
    tensor2d.set(1, 1, 3.0);
    let tensor2d_json = serde_json::to_value(&tensor2d).expect("tensor2d should serialize");
    assert_eq!(tensor2d_json["kind"], "tensor_2d");
    let tensor2d_restored: Tensor2D<f64> =
        serde_json::from_value(tensor2d_json).expect("tensor2d should deserialize");
    assert_eq!(tensor2d_restored.get(0, 0), 1.0);
    assert_eq!(tensor2d_restored.get(1, 1), 3.0);

    let mut matrix = Matrix::<f64>::empty(2, 3);
    matrix.set(0, 1, 2.0);
    matrix.set(1, 2, 4.0);
    let matrix_json = serde_json::to_value(&matrix).expect("matrix should serialize");
    assert_eq!(matrix_json["kind"], "matrix");
    let matrix_restored: Matrix<f64> =
        serde_json::from_value(matrix_json).expect("matrix should deserialize");
    assert_eq!(matrix_restored.get(0, 1), 2.0);
    assert_eq!(matrix_restored.get(1, 2), 4.0);

    let mut vector_list = VectorList::<f64>::empty(3, 2);
    vector_list.set(0, 0, 1.0);
    vector_list.set(1, 2, 5.0);
    let vector_list_json =
        serde_json::to_value(&vector_list).expect("vector list should serialize");
    assert_eq!(vector_list_json["kind"], "vector_list");
    let vector_list_restored: VectorList<f64> =
        serde_json::from_value(vector_list_json).expect("vector list should deserialize");
    assert_eq!(vector_list_restored.get(0, 0), 1.0);
    assert_eq!(vector_list_restored.get(1, 2), 5.0);
}

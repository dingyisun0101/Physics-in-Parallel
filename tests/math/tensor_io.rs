use physics_in_parallel::math::tensor::{Dense, Sparse, Tensor};

#[test]
fn dense_tensor_streaming_json_roundtrip_preserves_flat_schema() {
    let tensor = Tensor::<i64, Dense>::from_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let mut bytes = Vec::new();

    serde_json::to_writer(&mut bytes, &tensor).expect("serialize dense tensor to JSON writer");

    let value: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse serialized dense tensor");
    assert_eq!(
        value,
        serde_json::json!({
            "kind": "tensor",
            "shape": [2, 3],
            "data": [1, 2, 3, 4, 5, 6]
        })
    );

    let decoded: Tensor<i64, Dense> =
        serde_json::from_slice(&bytes).expect("deserialize dense tensor");
    assert_eq!(decoded.shape(), tensor.shape());
    for row in 0..2 {
        for column in 0..3 {
            assert_eq!(decoded.get(&[row, column]), tensor.get(&[row, column]));
        }
    }
}

#[test]
fn sparse_tensor_roundtrip_preserves_existing_dense_json_schema() {
    let tensor =
        Tensor::<i64, Sparse>::from_triplets(vec![2, 3], vec![(vec![0, 1], 7), (vec![1, 2], -4)]);

    let bytes = serde_json::to_vec(&tensor).expect("serialize sparse tensor");
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse serialized sparse tensor");
    assert_eq!(value["kind"], "tensor_sparse");
    assert_eq!(value["shape"], serde_json::json!([2, 3]));
    assert_eq!(value["data"], serde_json::json!([0, 7, 0, 0, 0, -4]));

    let decoded: Tensor<i64, Sparse> =
        serde_json::from_slice(&bytes).expect("deserialize sparse tensor");
    assert_eq!(decoded.shape(), tensor.shape());
    assert_eq!(decoded.nnz(), 2);
    assert_eq!(decoded.get(&[0, 1]), 7);
    assert_eq!(decoded.get(&[1, 2]), -4);
}

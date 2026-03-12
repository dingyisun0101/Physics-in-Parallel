use ndarray::arr2;
use physics_in_parallel::math::tensor::core::{dense, sparse, Dense, Sparse, Tensor};
use physics_in_parallel::math::tensor::core::tensor_trait::TensorTrait;
use physics_in_parallel::math::tensor::{Matrix, Tensor2D, VectorList};

#[test]
fn dense_tensor_uses_flat_schema() {
    let mut t = dense::Tensor::<f64>::empty(&[2, 3]);
    t.set(&[0, 0], 1.0);
    t.set(&[1, 2], 9.0);

    let v = serde_json::to_value(&t).expect("serialize dense tensor");
    assert_eq!(v["kind"], "tensor");
    assert_eq!(v["shape"], serde_json::json!([2, 3]));
    assert_eq!(v["data"], serde_json::json!([1.0, 0.0, 0.0, 0.0, 0.0, 9.0]));
    assert_eq!(v.as_object().expect("json object").len(), 3);

    let back: dense::Tensor<f64> = serde_json::from_value(v).expect("deserialize dense tensor");
    assert_eq!(back.shape, vec![2, 3]);
    assert_eq!(back.get(&[0, 0]), 1.0);
    assert_eq!(back.get(&[1, 2]), 9.0);
}

#[test]
fn sparse_tensor_serializes_as_dense_payload() {
    let mut s = sparse::Tensor::<f64>::empty(&[2, 3]);
    s.set(&[0, 1], 2.0);
    s.set(&[1, 2], -5.0);

    let v = serde_json::to_value(&s).expect("serialize sparse tensor");
    assert_eq!(v["kind"], "tensor_sparse");
    assert_eq!(v["shape"], serde_json::json!([2, 3]));
    assert_eq!(v["data"], serde_json::json!([0.0, 2.0, 0.0, 0.0, 0.0, -5.0]));

    let back: sparse::Tensor<f64> = serde_json::from_value(v).expect("deserialize sparse tensor");
    assert_eq!(back.shape(), &[2, 3]);
    assert_eq!(back.get(&[0, 1]), 2.0);
    assert_eq!(back.get(&[1, 2]), -5.0);
    assert_eq!(back.nnz(), 2);
}

#[test]
fn tensor2d_and_matrix_use_flat_row_major_data() {
    let mut t2 = Tensor2D::<f64>::empty(2, 2);
    t2.set(0, 0, 1.0);
    t2.set(0, 1, 2.0);
    t2.set(1, 0, 3.0);
    t2.set(1, 1, 4.0);

    let t2_json = serde_json::to_value(&t2).expect("serialize tensor2d");
    assert_eq!(t2_json["kind"], "tensor_2d");
    assert_eq!(t2_json["shape"], serde_json::json!([2, 2]));
    assert_eq!(t2_json["data"], serde_json::json!([1.0, 2.0, 3.0, 4.0]));

    let mut m = Matrix::<f64>::empty(2, 2);
    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(1, 0, 3.0);
    m.set(1, 1, 4.0);

    let m_json = serde_json::to_value(&m).expect("serialize matrix");
    assert_eq!(m_json["kind"], "matrix");
    assert_eq!(m_json["shape"], serde_json::json!([2, 2]));
    assert_eq!(m_json["data"], serde_json::json!([1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn vector_list_schema_is_n_dim_and_ndarray_matches() {
    let mut vl = VectorList::<f64>::empty(3, 2);
    vl.set(0, 0, 1.0);
    vl.set(0, 1, 2.0);
    vl.set(0, 2, 3.0);
    vl.set(1, 0, 4.0);
    vl.set(1, 1, 5.0);
    vl.set(1, 2, 6.0);

    let v = serde_json::to_value(&vl).expect("serialize vector list");
    assert_eq!(v["kind"], "vector_list");
    assert_eq!(v["shape"], serde_json::json!([2, 3]));
    assert_eq!(v["data"], serde_json::json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

    let back: VectorList<f64> = serde_json::from_value(v).expect("deserialize vector list");
    assert_eq!(back.shape(), [2, 3]);
    assert_eq!(back.get_vector(0), [1.0, 2.0, 3.0]);
    assert_eq!(back.get_vector(1), [4.0, 5.0, 6.0]);

    let arr = back.to_ndarray();
    assert_eq!(arr.shape(), &[2, 3]);
    assert_eq!(arr, arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
}

#[test]
fn unified_tensor_kinds_follow_backend() {
    let mut d = Tensor::<f64, Dense>::empty(&[2]);
    d.set(&[1], 3.0);
    let dense_json = serde_json::to_value(&d).expect("serialize unified dense");
    assert_eq!(dense_json["kind"], "tensor");

    let s = d.to_sparse();
    let sparse_json = serde_json::to_value(&s).expect("serialize unified sparse");
    assert_eq!(sparse_json["kind"], "tensor_sparse");

    let back_dense: Tensor<f64, Dense> =
        serde_json::from_value(dense_json).expect("deserialize unified dense");
    let back_sparse: Tensor<f64, Sparse> =
        serde_json::from_value(sparse_json).expect("deserialize unified sparse");
    assert_eq!(back_dense.get(&[1]), 3.0);
    assert_eq!(back_sparse.get(&[1]), 3.0);
}

#[test]
fn malformed_payloads_fail_with_clear_errors() {
    let bad_dense = serde_json::json!({
        "kind": "tensor",
        "shape": [2, 2],
        "data": [1.0, 2.0, 3.0]
    });
    let err = serde_json::from_value::<dense::Tensor<f64>>(bad_dense)
        .expect_err("dense payload length mismatch must fail")
        .to_string();
    assert!(err.contains("data length mismatch"));

    let bad_kind = serde_json::json!({
        "kind": "tensor",
        "shape": [2, 2],
        "data": [0.0, 0.0, 0.0, 0.0]
    });
    let err = serde_json::from_value::<sparse::Tensor<f64>>(bad_kind)
        .expect_err("sparse kind mismatch must fail")
        .to_string();
    assert!(err.contains("tensor_sparse kind"));
}

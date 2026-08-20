use ndarray::array;
use serde_json::json;

use physics_in_parallel::prelude::advanced::{NdarrayConvert, SparseMatrix};
use physics_in_parallel::prelude::basic::{DenseMatrix, Matrix};

#[test]
fn dense_matrix_serializes_as_flat_matrix_payload() {
    let matrix = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let value = matrix.serialize_value().expect("serialize matrix value");

    assert_eq!(value["kind"], "matrix");
    assert_eq!(value["version"], 1);
    assert_eq!(value["scalar"], "f64");
    assert_eq!(value["shape"], json!([2, 3]));
    assert_eq!(value["data"], json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert_eq!(value.as_object().expect("matrix object").len(), 5);
}

#[test]
fn dense_matrix_json_round_trip_preserves_entries() {
    let matrix = Matrix::<i64>::from_vec(2, 2, vec![1, -2, 3, 4]);
    let json = matrix.serialize().expect("serialize matrix string");

    let decoded: DenseMatrix<i64> = serde_json::from_str(&json).expect("deserialize matrix");

    assert_eq!(decoded.shape(), [2, 2]);
    assert_eq!(decoded.get(0, 0), 1);
    assert_eq!(decoded.get(0, 1), -2);
    assert_eq!(decoded.get(1, 0), 3);
    assert_eq!(decoded.get(1, 1), 4);
}

#[test]
fn sparse_matrix_serializes_as_compact_sparse_payload_and_round_trips() {
    let sparse = SparseMatrix::<i64>::from_triplets(2, 3, [(0, 1, 5), (1, 2, -7)]);

    let value = sparse.serialize_value().expect("serialize sparse matrix");
    assert_eq!(value["kind"], "matrix_sparse");
    assert_eq!(value["version"], 1);
    assert_eq!(value["scalar"], "i64");
    assert_eq!(value["shape"], json!([2, 3]));
    assert_eq!(value["indices"], json!([1, 5]));
    assert_eq!(value["values"], json!([5, -7]));
    assert!(value.get("data").is_none());

    let decoded: SparseMatrix<i64> =
        serde_json::from_value(value).expect("deserialize sparse matrix");
    assert_eq!(decoded.shape(), [2, 3]);
    assert_eq!(decoded.get(0, 1), 5);
    assert_eq!(decoded.get(1, 2), -7);
    assert_eq!(decoded.get(0, 0), 0);
}

#[test]
fn matrix_deserialization_rejects_non_matrix_payloads() {
    let err = serde_json::from_value::<DenseMatrix<i64>>(json!({
        "kind": "tensor",
        "version": 1,
        "scalar": "i64",
        "shape": [2, 2],
        "data": [1, 2, 3, 4]
    }))
    .unwrap_err();

    assert!(err.to_string().contains("matrix kind must be 'matrix'"));
}

#[test]
fn matrix_deserialization_rejects_non_rank_two_shape() {
    let err = serde_json::from_value::<DenseMatrix<i64>>(json!({
        "kind": "matrix",
        "version": 1,
        "scalar": "i64",
        "shape": [4],
        "data": [1, 2, 3, 4]
    }))
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("matrix shape rank mismatch: expected 2, got 1")
    );
}

#[test]
fn dense_matrix_converts_to_and_from_ndarray() {
    let array = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let matrix = Matrix::<f64>::from_ndarray(&array);
    assert_eq!(matrix.shape(), [2, 3]);
    assert_eq!(matrix.get(1, 2), 6.0);

    let round_trip = matrix.to_ndarray();
    assert_eq!(round_trip, array);

    let via_trait = <DenseMatrix<f64> as NdarrayConvert>::from_ndarray(&array);
    assert_eq!(via_trait.to_ndarray(), array);
}

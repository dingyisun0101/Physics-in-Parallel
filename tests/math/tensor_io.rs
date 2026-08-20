use physics_in_parallel::prelude::advanced::{Dense, Sparse};
use physics_in_parallel::prelude::basic::{Complex, Tensor};

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
            "version": 1,
            "scalar": "i64",
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
fn sparse_tensor_roundtrip_is_compact_deterministic_and_direct() {
    let tensor =
        Tensor::<i64, Sparse>::from_triplets(vec![2, 3], vec![(vec![0, 1], 7), (vec![1, 2], -4)]);

    let bytes = serde_json::to_vec(&tensor).expect("serialize sparse tensor");
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse serialized sparse tensor");
    assert_eq!(value["kind"], "tensor_sparse");
    assert_eq!(value["version"], 1);
    assert_eq!(value["scalar"], "i64");
    assert_eq!(value["shape"], serde_json::json!([2, 3]));
    assert_eq!(value["indices"], serde_json::json!([1, 5]));
    assert_eq!(value["values"], serde_json::json!([7, -4]));
    assert!(value.get("data").is_none());

    let decoded: Tensor<i64, Sparse> =
        serde_json::from_slice(&bytes).expect("deserialize sparse tensor");
    assert_eq!(decoded.shape(), tensor.shape());
    assert_eq!(decoded.nnz(), 2);
    assert_eq!(decoded.get(&[0, 1]), 7);
    assert_eq!(decoded.get(&[1, 2]), -4);
}

#[test]
fn sparse_tensor_output_size_tracks_nonzeros_not_logical_size() {
    let tensor = Tensor::<i64, Sparse>::from_triplets(
        vec![1_000, 1_000],
        vec![(vec![0, 1], 7), (vec![999, 999], -4)],
    );
    let bytes = serde_json::to_vec(&tensor).unwrap();
    assert!(
        bytes.len() < 180,
        "unexpected sparse JSON size: {}",
        bytes.len()
    );
    assert!(!bytes.windows(7).any(|window| window == b"\"data\":"));
}

#[test]
fn tensor_json_roundtrip_preserves_sensitive_float_bits_and_rejects_nonfinite_values() {
    let sensitive = -0.135_508_183_292_136_99_f64;
    let tensor = Tensor::<f64, Dense>::from_vec(&[1], vec![sensitive]);
    let bytes = serde_json::to_vec(&tensor).unwrap();
    let decoded: Tensor<f64, Dense> = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded.get(&[0]).to_bits(), sensitive.to_bits());

    let invalid = Tensor::<f64, Dense>::from_vec(&[1], vec![f64::NAN]);
    let error = serde_json::to_vec(&invalid).unwrap_err();
    assert!(error.to_string().contains("non-finite scalar"));
}

#[test]
fn sparse_tensor_decoder_rejects_noncanonical_or_invalid_entries() {
    for (payload, expected) in [
        (
            serde_json::json!({
                "kind": "tensor_sparse", "version": 1, "scalar": "i64", "shape": [4],
                "indices": [2, 1], "values": [7, 8]
            }),
            "strictly increasing",
        ),
        (
            serde_json::json!({
                "kind": "tensor_sparse", "version": 1, "scalar": "i64", "shape": [4],
                "indices": [4], "values": [7]
            }),
            "out of bounds",
        ),
        (
            serde_json::json!({
                "kind": "tensor_sparse", "version": 1, "scalar": "i64", "shape": [4],
                "indices": [1], "values": [0]
            }),
            "explicit zero",
        ),
        (
            serde_json::json!({
                "kind": "tensor_sparse", "version": 2, "scalar": "i64", "shape": [4],
                "indices": [1], "values": [7]
            }),
            "version mismatch",
        ),
        (
            serde_json::json!({
                "kind": "tensor_sparse", "version": 1, "scalar": "f64", "shape": [4],
                "indices": [1], "values": [7]
            }),
            "scalar mismatch",
        ),
    ] {
        let error = serde_json::from_value::<Tensor<i64, Sparse>>(payload).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?} in {error}"
        );
    }
}

#[test]
fn complex_tensor_schema_has_stable_scalar_metadata_and_round_trips() {
    let tensor = Tensor::<Complex<f64>, Dense>::from_vec(
        &[2],
        vec![Complex::new(1.5, -2.0), Complex::new(0.0, 3.0)],
    );
    let value = serde_json::to_value(&tensor).unwrap();
    assert_eq!(value["scalar"], "complex_f64");
    assert_eq!(value["data"][0], serde_json::json!([1.5, -2.0]));
    let decoded: Tensor<Complex<f64>, Dense> = serde_json::from_value(value).unwrap();
    assert_eq!(decoded.get(&[0]), Complex::new(1.5, -2.0));
    assert_eq!(decoded.get(&[1]), Complex::new(0.0, 3.0));
}

//! Independent wire fixtures: expected documents never come from PiP serializers.
use super::support::{BACKENDS, Report};
use physics_in_parallel::prelude::advanced::*;
use physics_in_parallel::prelude::basic::*;
use physics_in_parallel::prelude::models::PhysObj;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    backend: String,
    tensor: Box<serde_json::value::RawValue>,
}
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Dense<T> {
    kind: String,
    version: u32,
    scalar: String,
    shape: Vec<usize>,
    data: Vec<T>,
}
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Sparse<T> {
    kind: String,
    version: u32,
    scalar: String,
    shape: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<T>,
}

fn fixture<T: Serialize>(
    backend: Backend,
    scalar: &str,
    values: &[T],
    stored: &[(usize, T)],
) -> String {
    let payload = if backend == Backend::Dense {
        format!(
            r#"{{"kind":"tensor","version":2,"scalar":"{scalar}","shape":[2,3],"data":{}}}"#,
            serde_json::to_string(values).unwrap()
        )
    } else {
        format!(
            r#"{{"kind":"tensor_sparse","version":2,"scalar":"{scalar}","shape":[2,3],"indices":{},"values":{}}}"#,
            serde_json::to_string(&stored.iter().map(|x| x.0).collect::<Vec<_>>()).unwrap(),
            serde_json::to_string(&stored.iter().map(|x| &x.1).collect::<Vec<_>>()).unwrap()
        )
    };
    format!(r#"{{"backend":"{backend}","tensor":{payload}}}"#)
}

fn scalar_fixture<T: Scalar + Serialize + DeserializeOwned>(
    report: &mut Report,
    scalar: &str,
    values: Vec<T>,
) {
    let stored: Vec<_> = values
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != T::zero())
        .map(|(i, &v)| (i, v))
        .collect();
    for backend in BACKENDS {
        let reference = fixture(backend, scalar, &values, &stored);
        // Compare all semantic wrappers directly to an external document.
        let tensor: Tensor<T> = serde_json::from_str(&reference).unwrap();
        let matrix: Matrix<T> = serde_json::from_str(&reference).unwrap();
        let vectors: VectorList<T> = serde_json::from_str(&reference).unwrap();
        for actual in [
            tensor.values().collect::<Vec<_>>(),
            matrix.values().collect(),
            vectors.values().collect(),
        ] {
            report.exact(
                &format!("JSON.{scalar}.decode.{backend}"),
                actual,
                values.clone(),
            );
        }
        assert_eq!(tensor.shape(), [2, 3]);
        assert_eq!(matrix.shape(), [2, 3]);
        assert_eq!(vectors.shape(), [2, 3]);
        // Insert sparse entries backwards to verify canonical output order.
        let tensor = Tensor::from_entries(
            &[2, 3],
            backend,
            stored.iter().rev().map(|&(i, v)| (vec![i / 3, i % 3], v)),
        )
        .unwrap();
        let matrix = Matrix::from_values(2, 3, backend, values.clone()).unwrap();
        let vectors = VectorList::from_values(3, 2, backend, values.clone()).unwrap();
        let documents = [
            serde_json::to_string(&tensor).unwrap(),
            serde_json::to_string(&matrix).unwrap(),
            serde_json::to_string(&vectors).unwrap(),
        ];
        let expected: Envelope = serde_json::from_str(&reference).unwrap();
        for document in documents {
            let actual: Envelope = serde_json::from_str(&document).unwrap();
            report.exact("JSON.backend tag", actual.backend, expected.backend.clone());
            if backend == Backend::Dense {
                report.exact(
                    &format!("JSON.{scalar}.encode.dense"),
                    serde_json::from_str::<Dense<T>>(actual.tensor.get()).unwrap(),
                    serde_json::from_str::<Dense<T>>(expected.tensor.get()).unwrap(),
                );
            } else {
                report.exact(
                    &format!("JSON.{scalar}.encode.sparse"),
                    serde_json::from_str::<Sparse<T>>(actual.tensor.get()).unwrap(),
                    serde_json::from_str::<Sparse<T>>(expected.tensor.get()).unwrap(),
                );
            }
        }
        // JSON member order must not change scalar precision or accepted types.
        let reordered = format!(
            r#"{{"tensor":{},"backend":"{backend}"}}"#,
            expected.tensor.get()
        );
        let restored: Tensor<T> = serde_json::from_str(&reordered).unwrap();
        report.exact(
            &format!("JSON.{scalar}.field_order"),
            restored.values().collect::<Vec<_>>(),
            values.clone(),
        );
        let matrix: Matrix<T> = serde_json::from_str(&reordered).unwrap();
        let vectors: VectorList<T> = serde_json::from_str(&reordered).unwrap();
        report.exact(
            "JSON.Matrix reordered payload",
            matrix.values().collect::<Vec<_>>(),
            values.clone(),
        );
        report.exact(
            "JSON.VectorList reordered payload",
            vectors.values().collect::<Vec<_>>(),
            values.clone(),
        );
    }
}

#[test]
fn every_scalar_and_container_matches_independent_wire_fixtures() {
    let mut report = Report::default();
    macro_rules! integers {
        ($($ty:ty),*) => {$(scalar_fixture::<$ty>(&mut report,stringify!($ty),vec![0,1,<$ty>::MIN,<$ty>::MAX,0,17]);)*};
    }
    integers!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
    );
    scalar_fixture::<f32>(
        &mut report,
        "f32",
        vec![
            0.0,
            1.25,
            -2.75,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MAX,
        ],
    );
    scalar_fixture::<f64>(
        &mut report,
        "f64",
        vec![
            0.0,
            1.25,
            -2.75,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::MAX,
        ],
    );
    scalar_fixture::<Complex<f32>>(
        &mut report,
        "complex_f32",
        vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.25, -2.75),
            Complex::new(-3.5, 0.0),
            Complex::new(0.0, 4.5),
            Complex::new(f32::MIN_POSITIVE, f32::from_bits(1)),
            Complex::new(f32::MAX, -f32::MAX),
        ],
    );
    scalar_fixture::<Complex<f64>>(
        &mut report,
        "complex_f64",
        vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.25, -2.75),
            Complex::new(-3.5, 0.0),
            Complex::new(0.0, 4.5),
            Complex::new(f64::MIN_POSITIVE, f64::from_bits(1)),
            Complex::new(f64::MAX, -f64::MAX),
        ],
    );
    report.finish();
}

#[test]
fn dense_float_serialization_preserves_bits_including_negative_zero() {
    let mut report = Report::default();
    macro_rules! floats {
        ($ty:ty) => {{
            let input = vec![
                0.0,
                -0.0,
                <$ty>::from_bits(1),
                -<$ty>::from_bits(1),
                <$ty>::MIN_POSITIVE,
                <$ty>::MAX,
                1.0 / 3.0,
            ];
            let tensor =
                Tensor::from_values(&[input.len()], Backend::Dense, input.clone()).unwrap();
            let restored: Tensor<$ty> =
                serde_json::from_str(&serde_json::to_string(&tensor).unwrap()).unwrap();
            report.exact(
                concat!("JSON.bitwise.", stringify!($ty)),
                restored.values().map(|x| x.to_bits()).collect::<Vec<_>>(),
                input.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            );
        }};
    }
    floats!(f32);
    floats!(f64);
    report.finish();
}

fn rejects<T: DeserializeOwned>(report: &mut Report, label: &str, document: &Value) {
    let encoded = serde_json::to_string(document).unwrap();
    let decoded = std::panic::catch_unwind(|| serde_json::from_str::<T>(&encoded));
    assert!(
        decoded.is_ok(),
        "{label}: deserializer panicked for {encoded}"
    );
    assert!(
        decoded.unwrap().is_err(),
        "{label}: accepted invalid document {encoded}"
    );
    report.exact(label, true, true);
}

#[test]
fn malformed_container_payloads_return_errors_without_panicking() {
    let mut report = Report::default();
    for backend in BACKENDS {
        let valid: Value = serde_json::from_str(&fixture(
            backend,
            "f64",
            &[0.0, 2.0, 0.0, -3.0, 0.0, 0.0],
            &[(1, 2.0), (3, -3.0)],
        ))
        .unwrap();
        let fields = [
            ("version", json!(0)),
            ("version", json!(3)),
            ("kind", json!("matrix")),
            ("scalar", json!("f32")),
            ("scalar", json!("unknown")),
            ("shape", json!([])),
            ("shape", json!([2, 0])),
            ("shape", json!([-1, 3])),
            ("shape", json!([usize::MAX, 2])),
            ("unexpected", json!(true)),
        ];
        for (field, value) in fields {
            let mut bad = valid.clone();
            bad["tensor"][field] = value;
            rejects::<Tensor<f64>>(
                &mut report,
                &format!("JSON.invalid.{backend}.{field}"),
                &bad,
            );
            rejects::<Matrix<f64>>(&mut report, "JSON.invalid.Matrix", &bad);
            rejects::<VectorList<f64>>(&mut report, "JSON.invalid.VectorList", &bad);
        }
        for field in ["version", "kind", "scalar", "shape"] {
            let mut bad = valid.clone();
            bad["tensor"].as_object_mut().unwrap().remove(field);
            rejects::<Tensor<f64>>(&mut report, "JSON.missing field", &bad);
        }
        let edits = if backend == Backend::Dense {
            vec![
                ("data", json!([1.0])),
                ("data", json!([0, 2, 0, -3, 0, null])),
                ("data", json!([0, 2, 0, -3, 0, "NaN"])),
                ("data", Value::Null),
                ("indices", json!([])),
                ("indices", Value::Null),
                ("values", json!([])),
            ]
        } else {
            vec![
                ("indices", json!([1, 1])),
                ("indices", json!([3, 1])),
                ("indices", json!([1, 6])),
                ("indices", json!([-1, 3])),
                ("indices", json!([1])),
                ("values", json!([2.0])),
                ("values", json!([2.0, 0.0])),
                ("values", json!([2.0, null])),
                ("indices", Value::Null),
                ("values", Value::Null),
                ("data", json!([])),
                ("data", Value::Null),
            ]
        };
        for (field, value) in edits {
            let mut bad = valid.clone();
            bad["tensor"][field] = value;
            rejects::<Tensor<f64>>(
                &mut report,
                &format!("JSON.invalid.{backend}.{field}"),
                &bad,
            );
        }
        for field in if backend == Backend::Dense {
            vec!["data"]
        } else {
            vec!["indices", "values"]
        } {
            let mut missing = valid.clone();
            missing["tensor"].as_object_mut().unwrap().remove(field);
            rejects::<Tensor<f64>>(&mut report, "JSON.missing storage array", &missing);
        }
        let mut wrong_rank = valid.clone();
        wrong_rank["tensor"]["shape"] = json!([6]);
        assert!(serde_json::from_value::<Tensor<f64>>(wrong_rank.clone()).is_ok());
        rejects::<Matrix<f64>>(&mut report, "JSON.Matrix rank", &wrong_rank);
        rejects::<VectorList<f64>>(&mut report, "JSON.VectorList rank", &wrong_rank);
    }
    // Product fits usize but exceeds the signed index space required by storage.
    for shape in [vec![isize::MAX as usize + 1], vec![isize::MAX as usize, 2]] {
        let bad = json!({"backend":"sparse","tensor":{"kind":"tensor_sparse","version":2,"scalar":"f64","shape":shape,"indices":[],"values":[]}});
        rejects::<Tensor<f64>>(&mut report, "JSON.signed index overflow", &bad);
    }
    report.finish();
}

#[test]
fn nonfinite_values_are_rejected_at_serialization_boundaries() {
    let mut report = Report::default();
    for backend in BACKENDS {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let tensor = Tensor::from_values(&[2], backend, vec![1.0, value]).unwrap();
            let matrix = Matrix::from_values(1, 2, backend, vec![1.0, value]).unwrap();
            let vectors = VectorList::from_values(2, 1, backend, vec![1.0, value]).unwrap();
            for error in [
                serde_json::to_string(&tensor).unwrap_err(),
                serde_json::to_string(&matrix).unwrap_err(),
                serde_json::to_string(&vectors).unwrap_err(),
            ] {
                assert!(error.to_string().contains("non-finite"));
                report.exact("JSON.reject nonfinite real", true, true);
            }
            for z in [Complex::new(value, 0.0), Complex::new(0.0, value)] {
                let tensor = Tensor::from_values(&[1], backend, vec![z]).unwrap();
                assert!(serde_json::to_string(&tensor).is_err());
                report.exact("JSON.reject nonfinite complex", true, true);
            }
        }
    }
    report.finish();
}

#[test]
fn huge_sparse_wire_payload_preserves_only_explicit_entries() {
    let mut report = Report::default();
    let document = json!({"backend":"sparse","tensor":{"kind":"tensor_sparse","version":2,"scalar":"f64","shape":[1000000000],"indices":[7,999999999],"values":[1.25,-3.5]}});
    let t: Tensor<f64> = serde_json::from_value(document.clone()).unwrap();
    report.exact(
        "JSON.huge sparse stored entries",
        t.stored_entries()
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>(),
        [(7, 1.25), (999999999, -3.5)].into_iter().collect(),
    );
    report.exact(
        "JSON.huge sparse encoding",
        serde_json::to_value(&t).unwrap(),
        document,
    );
    assert_eq!(t.get(&[1]).unwrap(), 0.0);
    report.finish();
}

#[test]
fn heterogeneous_attributes_decode_external_documents_and_keep_slot_ids() {
    let mut report = Report::default();
    let r: Value = serde_json::from_str(&fixture(
        Backend::Dense,
        "f64",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[],
    ))
    .unwrap();
    let codes: Value = serde_json::from_str(&fixture(
        Backend::Sparse,
        "u8",
        &[0, 1, 0, 2, 0, 3],
        &[(1, 1), (3, 2), (5, 3)],
    ))
    .unwrap();
    let document = json!({"kind":"phys_obj","version":2,"meta":{"id":17,"label":"fixture","comment":"external"},"core":{"kind":"attrs_core","version":2,"n_objects":2,"num_attrs":2,"slot_count":4,"attrs":[{"id":1,"label":"r","scalar":"f64","payload":r},{"id":3,"label":"codes","scalar":"u8","payload":codes}]}});
    let object: PhysObj = serde_json::from_value(document.clone()).unwrap();
    report.exact(
        "JSON.PhysObj metadata",
        (object.id(), object.label(), object.comment()),
        (17, "fixture", "external"),
    );
    report.exact(
        "JSON.PhysObj f64 values",
        object
            .attribute::<f64>("r")
            .unwrap()
            .values()
            .collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    report.exact(
        "JSON.PhysObj u8 values",
        object
            .attribute::<u8>("codes")
            .unwrap()
            .values()
            .collect::<Vec<_>>(),
        vec![0, 1, 0, 2, 0, 3],
    );
    report.exact(
        "JSON.PhysObj ids",
        (
            object.attributes().id_of("r").unwrap(),
            object.attributes().id_of("codes").unwrap(),
        ),
        (1, 3),
    );
    report.exact(
        "JSON.PhysObj encoding",
        serde_json::to_value(&object).unwrap(),
        document.clone(),
    );
    for (pointer, value) in [
        ("/version", json!(1)),
        ("/kind", json!("tensor")),
        ("/core/version", json!(1)),
        ("/core/n_objects", json!(3)),
        ("/core/num_attrs", json!(3)),
        ("/core/slot_count", json!(1)),
        ("/core/attrs/1/id", json!(1)),
        ("/core/attrs/1/label", json!("r")),
        ("/core/attrs/1/scalar", json!("f64")),
        ("/core/attrs/1/scalar", json!("missing")),
    ] {
        let mut bad = document.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        rejects::<PhysObj>(&mut report, &format!("JSON.invalid.PhysObj{pointer}"), &bad);
    }
    report.finish();
}

fn decode_like<T: DeserializeOwned>(_: &T, value: &Value) -> Result<T, serde_json::Error> {
    serde_json::from_value(value.clone())
}

#[test]
fn structured_matrix_json_matches_logical_dense_and_sparse_references() {
    let mut report = Report::default();
    macro_rules! structured {
        ($kind:ident,$support:expr,$mirror:expr) => {{
            let n = 5;
            let mut matrix = $kind::<f64>::zeros(n,n);
            let mut values = vec![0.0;n*n];
            for i in 0..n {for j in 0..n {
                if ($support)(i,j) {
                    let value = (i*n+j+1) as f64 / 4.0;
                    matrix.set(i as isize,j as isize,value);
                    values[i*n+j] = value;
                    if $mirror != 0 && i!=j {values[j*n+i] = value*$mirror as f64;}
                }
            }}
            let expected = json!({"kind":"matrix","version":2,"scalar":"f64","shape":[n,n],"data":values});
            report.exact(concat!("JSON.structured.",stringify!($kind)),serde_json::to_value(&matrix).unwrap(),expected.clone());
            let dense = matrix.to_dense_matrix();
            let decoded = decode_like(&dense,&expected).unwrap();
            report.exact("JSON.advanced dense decode",(0..n*n).map(|i|decoded.get_flat(i as isize)).collect::<Vec<_>>(),values.clone());
            let indices: Vec<_> = values.iter().enumerate().filter(|(_,v)|**v!=0.0).map(|(i,_)|i).collect();
            let stored: Vec<_> = indices.iter().map(|&i|values[i]).collect();
            let expected = json!({"kind":"matrix_sparse","version":2,"scalar":"f64","shape":[n,n],"indices":indices,"values":stored});
            let sparse = dense.to_sparse();
            report.exact("JSON.advanced sparse encode",serde_json::to_value(&sparse).unwrap(),expected.clone());
            let decoded = decode_like(&sparse,&expected).unwrap();
            report.exact("JSON.advanced sparse decode",(0..n*n).map(|i|decoded.get_flat(i as isize)).collect::<Vec<_>>(),values);
            let invalid = json!({"kind":"matrix_sparse","version":2,"scalar":"f64","shape":[isize::MAX as usize,2],"indices":[],"values":[]});
            assert!(std::panic::catch_unwind(||decode_like(&sparse,&invalid)).expect("oversized matrix panicked").is_err());
        }};
    }
    structured!(DiagonalMatrix, |i, j| i == j, 0);
    structured!(SymmetricMatrix, |i, j| i <= j, 1);
    structured!(AntiSymmetricMatrix, |i, j| i < j, -1);
    structured!(UpperTriangularMatrix, |i, j| i <= j, 0);
    structured!(StrictUpperTriangularMatrix, |i, j| i < j, 0);
    structured!(LowerTriangularMatrix, |i, j| i >= j, 0);
    structured!(StrictLowerTriangularMatrix, |i, j| i > j, 0);
    report.finish();
}

#[test]
fn duplicate_json_fields_and_crossed_backend_tags_are_rejected() {
    let mut report = Report::default();
    let document = fixture(Backend::Dense, "f64", &[0.0; 6], &[]);
    let bad_documents = [
        document.replacen(
            "\"backend\":\"dense\"",
            "\"backend\":\"dense\",\"backend\":\"sparse\"",
            1,
        ),
        document.replacen("\"version\":2", "\"version\":2,\"version\":2", 1),
        document.replacen("\"data\":", "\"data\":[],\"data\":", 1),
        document.replacen("\"dense\"", "\"sparse\"", 1),
        document.replacen("\"dense\"", "\"unknown\"", 1),
    ];
    for encoded in bad_documents {
        let result = std::panic::catch_unwind(|| serde_json::from_str::<Tensor<f64>>(&encoded));
        assert!(
            result.expect("invalid JSON panicked").is_err(),
            "accepted {encoded}"
        );
        report.exact("JSON.duplicate fields/backend mismatch", true, true);
    }
    report.finish();
}

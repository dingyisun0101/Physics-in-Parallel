use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use physics_in_parallel::space::discrete::square_lattice::{
    BoundaryCondition, SquareLattice, SquareLatticeConfig, SquareLatticeInitMethod,
};
use physics_in_parallel::space::io::square_lattice::save_square_lattice;

fn unique_tmp_json(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    env::temp_dir().join(format!("pip_{name}_{}_{}.json", std::process::id(), nanos))
}

#[test]
fn lattice_config_is_a_complete_validated_serde_boundary() {
    let config =
        SquareLatticeConfig::try_new(&[3, 5], BoundaryCondition::Reflective).expect("valid config");
    let value = serde_json::to_value(&config).expect("serialize config");
    assert_eq!(
        value,
        serde_json::json!({"shape": [3, 5], "boundary": "reflective"})
    );
    let roundtrip: SquareLatticeConfig = serde_json::from_value(value).expect("deserialize config");
    assert_eq!(roundtrip, config);

    for invalid in [
        serde_json::json!({"shape": [], "boundary": "periodic"}),
        serde_json::json!({"shape": [2, 0], "boundary": "periodic"}),
        serde_json::json!({"shape": [2], "boundary": "periodic", "extra": true}),
    ] {
        assert!(serde_json::from_value::<SquareLatticeConfig>(invalid).is_err());
    }
    assert!(SquareLatticeConfig::try_new(&[usize::MAX, 2], BoundaryCondition::Periodic).is_err());
}

#[test]
fn lattice_periodic_roundtrip_uses_lattice_schema() {
    let lattice = SquareLattice::<usize>::new(
        SquareLatticeConfig::new(&vec![3; 2], BoundaryCondition::Periodic),
        SquareLatticeInitMethod::Uniform { val: 7 },
    );

    let value = serde_json::to_value(&lattice).expect("serialize lattice");
    assert_eq!(value["kind"], "square_lattice_periodic");
    assert_eq!(value["version"], 1);
    assert_eq!(value["scalar"], "usize");
    assert_eq!(value["shape"], serde_json::json!([3, 3]));
    assert_eq!(value["data"].as_array().expect("data array").len(), 9);

    let back: SquareLattice<usize> = serde_json::from_value(value).expect("deserialize lattice");
    assert_eq!(back.cfg.shape(), [3, 3]);
    assert_eq!(back.cfg.boundary(), BoundaryCondition::Periodic);
    assert_eq!(back.data(), vec![7; 9].as_slice());
}

#[test]
fn lattice_reflective_roundtrip_uses_kind_tag() {
    let lattice = SquareLattice::<usize>::new(
        SquareLatticeConfig::new(&vec![2; 3], BoundaryCondition::Reflective),
        SquareLatticeInitMethod::Uniform { val: 1 },
    );

    let value = serde_json::to_value(&lattice).expect("serialize lattice");
    assert_eq!(value["kind"], "square_lattice_reflective");
    assert_eq!(value["version"], 1);
    assert_eq!(value["scalar"], "usize");
    assert_eq!(value["shape"], serde_json::json!([2, 2, 2]));

    let back: SquareLattice<usize> = serde_json::from_value(value).expect("deserialize lattice");
    assert_eq!(back.cfg.shape(), [2, 2, 2]);
    assert_eq!(back.cfg.boundary(), BoundaryCondition::Reflective);
}

#[test]
fn lattice_deserialize_rejects_bad_kind_and_shape() {
    let bad_kind = serde_json::json!({
        "kind": "grid_periodic",
        "version": 1,
        "scalar": "usize",
        "shape": [2, 2],
        "data": [0, 0, 0, 0]
    });
    let err = serde_json::from_value::<SquareLattice<usize>>(bad_kind)
        .expect_err("invalid kind must fail")
        .to_string();
    assert!(err.contains("square lattice kind"));

    let bad_shape = serde_json::json!({
        "kind": "square_lattice_periodic",
        "version": 1,
        "scalar": "usize",
        "shape": [2, 0],
        "data": []
    });
    let err = serde_json::from_value::<SquareLattice<usize>>(bad_shape)
        .expect_err("zero axis length must fail")
        .to_string();
    assert!(err.contains("nonzero"));

    let bad_len = serde_json::json!({
        "kind": "square_lattice_periodic",
        "version": 1,
        "scalar": "usize",
        "shape": [2, 2],
        "data": [0, 0, 0]
    });
    let err = serde_json::from_value::<SquareLattice<usize>>(bad_len)
        .expect_err("lattice len mismatch must fail")
        .to_string();
    assert!(err.contains("data length mismatch"));

    let wrong_scalar = serde_json::json!({
        "kind": "square_lattice_periodic",
        "version": 1,
        "scalar": "u64",
        "shape": [2, 2],
        "data": [1, 2, 3, 4]
    });
    let err = serde_json::from_value::<SquareLattice<usize>>(wrong_scalar)
        .expect_err("scalar mismatch must fail")
        .to_string();
    assert!(err.contains("scalar mismatch"));
}

#[test]
fn save_square_lattice_writes_flat_payload_schema() {
    let lattice = SquareLattice::<usize>::new(
        SquareLatticeConfig::new(&vec![4; 2], BoundaryCondition::Periodic),
        SquareLatticeInitMethod::Uniform { val: 9 },
    );

    let out = unique_tmp_json("save_square_lattice_flat_schema");
    save_square_lattice(&lattice, &vec![2; lattice.cfg.rank()], &out).expect("save lattice json");

    let raw = fs::read_to_string(&out).expect("read saved json");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("parse saved json");
    assert_eq!(value["kind"], "square_lattice_periodic");
    assert_eq!(value["version"], 1);
    assert_eq!(value["scalar"], "usize");
    assert_eq!(value["shape"], serde_json::json!([2, 2]));
    assert_eq!(value["data"].as_array().expect("data array").len(), 4);

    fs::remove_file(out).expect("cleanup saved json");
}

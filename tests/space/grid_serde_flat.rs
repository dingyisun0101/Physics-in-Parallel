use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use physics_in_parallel::space::discrete::representation::{save_grid, Grid, GridConfig, GridInitMethod};

fn unique_tmp_json(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    env::temp_dir().join(format!("pip_{name}_{}_{}.json", std::process::id(), nanos))
}

#[test]
fn grid_periodic_roundtrip_uses_new_schema() {
    let g = Grid::<usize>::new(
        GridConfig::new(2, 3, true),
        GridInitMethod::Uniform { val: 7 },
    );

    let v = serde_json::to_value(&g).expect("serialize grid");
    assert_eq!(v["kind"], "grid_periodic");
    assert_eq!(v["shape"], serde_json::json!([3, 3]));
    assert_eq!(v["data"].as_array().expect("data array").len(), 9);

    let back: Grid<usize> = serde_json::from_value(v).expect("deserialize grid");
    assert_eq!(back.cfg.d, 2);
    assert_eq!(back.cfg.l, 3);
    assert!(back.cfg.periodic);
    assert_eq!(back.data, vec![7; 9]);
}

#[test]
fn grid_clamped_roundtrip_uses_kind_tag() {
    let g = Grid::<usize>::new(
        GridConfig::new(3, 2, false),
        GridInitMethod::Uniform { val: 1 },
    );

    let v = serde_json::to_value(&g).expect("serialize grid");
    assert_eq!(v["kind"], "grid_clamped");
    assert_eq!(v["shape"], serde_json::json!([2, 2, 2]));

    let back: Grid<usize> = serde_json::from_value(v).expect("deserialize grid");
    assert_eq!(back.cfg.d, 3);
    assert_eq!(back.cfg.l, 2);
    assert!(!back.cfg.periodic);
}

#[test]
fn grid_deserialize_accepts_legacy_grid_kind_as_periodic() {
    let payload = serde_json::json!({
        "kind": "grid",
        "shape": [2, 2],
        "data": [1, 2, 3, 4]
    });

    let g: Grid<usize> = serde_json::from_value(payload).expect("deserialize legacy grid kind");
    assert_eq!(g.cfg.d, 2);
    assert_eq!(g.cfg.l, 2);
    assert!(g.cfg.periodic);
    assert_eq!(g.data, vec![1, 2, 3, 4]);
}

#[test]
fn grid_deserialize_rejects_bad_kind_and_shape() {
    let bad_kind = serde_json::json!({
        "kind": "grid_weird",
        "shape": [2, 2],
        "data": [0, 0, 0, 0]
    });
    let err = serde_json::from_value::<Grid<usize>>(bad_kind)
        .expect_err("invalid kind must fail")
        .to_string();
    assert!(err.contains("grid kind"));

    let bad_shape = serde_json::json!({
        "kind": "grid_periodic",
        "shape": [2, 3],
        "data": [0, 0, 0, 0, 0, 0]
    });
    let err = serde_json::from_value::<Grid<usize>>(bad_shape)
        .expect_err("non-cubic grid must fail")
        .to_string();
    assert!(err.contains("cubic"));

    let bad_len = serde_json::json!({
        "kind": "grid_periodic",
        "shape": [2, 2],
        "data": [0, 0, 0]
    });
    let err = serde_json::from_value::<Grid<usize>>(bad_len)
        .expect_err("grid len mismatch must fail")
        .to_string();
    assert!(err.contains("data length mismatch"));
}

#[test]
fn save_grid_writes_new_flat_payload_schema() {
    let g = Grid::<usize>::new(
        GridConfig::new(2, 4, true),
        GridInitMethod::Uniform { val: 9 },
    );

    let out = unique_tmp_json("save_grid_flat_schema");
    save_grid(&g, 2, &out).expect("save grid json");

    let raw = fs::read_to_string(&out).expect("read saved json");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse saved json");
    assert_eq!(v["kind"], "grid_periodic");
    assert_eq!(v["shape"], serde_json::json!([2, 2]));
    assert_eq!(v["data"].as_array().expect("data array").len(), 4);

    fs::remove_file(out).expect("cleanup saved json");
}

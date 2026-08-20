use physics_in_parallel::prelude::advanced::{AttrsCore, AttrsMeta, PhysObjAdvanced};
use physics_in_parallel::prelude::models::PhysObj;

#[test]
fn attrs_core_serializes_vector_payloads_in_flat_schema() {
    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 3, 2).expect("allocate r");
    core.allocate::<f64>("v", 3, 2).expect("allocate v");

    core.set_vector_of::<f64>("r", 0, &[1.0, 2.0, 3.0])
        .expect("set r[0]");
    core.set_vector_of::<f64>("r", 1, &[4.0, 5.0, 6.0])
        .expect("set r[1]");

    let json = core.serialize().expect("serialize attrs core");
    let value: serde_json::Value = serde_json::from_str(&json).expect("parse attrs core json");

    assert_eq!(value["kind"], "attrs_core");
    assert_eq!(value["version"], 1);
    assert_eq!(value["num_attrs"], 2);
    let attrs = value["attrs"].as_array().expect("attrs array");
    assert_eq!(attrs.len(), 2);

    for item in attrs {
        let payload = &item["payload"];
        assert_eq!(payload["kind"], "vector_list");
        assert_eq!(payload["version"], 1);
        assert_eq!(payload["scalar"], "f64");
        assert_eq!(payload["shape"], serde_json::json!([2, 3]));
        assert!(payload["data"].is_array());
        assert_eq!(payload.as_object().expect("payload object").len(), 5);
        assert_eq!(item["scalar"], "f64");
    }
}

#[test]
fn phys_obj_serialization_embeds_flat_vector_list_payloads() {
    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 2, 2).expect("allocate r");

    let obj = PhysObj::from_raw_parts(
        AttrsMeta {
            id: 42,
            label: "serde-flat".to_string(),
            comment: "schema check".to_string(),
        },
        core,
    );

    let json = obj.serialize().expect("serialize phys obj");
    let value: serde_json::Value = serde_json::from_str(&json).expect("parse phys obj json");

    assert_eq!(value["kind"], "phys_obj");
    assert_eq!(value["version"], 1);
    assert_eq!(value["meta"]["id"], 42);
    assert_eq!(value["core"]["num_attrs"], 1);
    assert_eq!(value["core"]["attrs"][0]["payload"]["kind"], "vector_list");
    assert_eq!(
        value["core"]["attrs"][0]["payload"]["shape"],
        serde_json::json!([2, 2])
    );
}

#[test]
fn phys_obj_roundtrip_restores_mixed_types_and_stable_attribute_ids() {
    let mut core = AttrsCore::empty();
    core.allocate::<f64>("position", 2, 2).unwrap();
    core.allocate::<i64>("species", 1, 2).unwrap();
    core.set_vector_of("position", 1, &[1.25, -2.5]).unwrap();
    core.set_vector_of("species", 0, &[7_i64]).unwrap();
    core.remove("position").unwrap();
    core.allocate::<f32>("velocity", 2, 2).unwrap();
    assert_eq!(core.id_of("species").unwrap(), 1);
    assert_eq!(core.id_of("velocity").unwrap(), 2);

    let original = PhysObj::from_raw_parts(AttrsMeta::new(9, "particles", "mixed"), core);
    let bytes = serde_json::to_vec(&original).unwrap();
    let decoded: PhysObj = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(decoded.metadata(), original.metadata());
    assert_eq!(decoded.attributes().id_of("species").unwrap(), 1);
    assert_eq!(decoded.attributes().id_of("velocity").unwrap(), 2);
    assert_eq!(
        decoded.attributes().vector_of::<i64>("species", 0).unwrap(),
        [7_i64]
    );
    assert_eq!(
        decoded
            .attributes()
            .vector_of::<f32>("velocity", 1)
            .unwrap(),
        [0.0, 0.0]
    );
}

#[test]
fn attrs_core_decoder_rejects_unknown_scalar_and_duplicate_ids() {
    let mut core = AttrsCore::empty();
    core.allocate::<f64>("position", 2, 1).unwrap();
    core.allocate::<i64>("species", 1, 1).unwrap();
    let value = serde_json::to_value(&core).unwrap();

    let mut unknown_scalar = value.clone();
    unknown_scalar["attrs"][0]["scalar"] = serde_json::json!("decimal128");
    let error = serde_json::from_value::<AttrsCore>(unknown_scalar).unwrap_err();
    assert!(error.to_string().contains("unsupported scalar kind"));

    let mut duplicate_id = value;
    duplicate_id["attrs"][1]["id"] = duplicate_id["attrs"][0]["id"].clone();
    let error = serde_json::from_value::<AttrsCore>(duplicate_id).unwrap_err();
    assert!(error.to_string().contains("duplicate attribute id"));
}

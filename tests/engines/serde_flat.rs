use physics_in_parallel::engines::soa::{AttrsCore, AttrsMeta, PhysObj};

#[test]
fn attrs_core_serializes_vector_payloads_in_flat_schema() {
    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 3, 2).expect("allocate r");
    core.allocate::<f64>("v", 3, 2).expect("allocate v");

    core.set_vector_of::<f64>("r", 0, &[1.0, 2.0, 3.0]).expect("set r[0]");
    core.set_vector_of::<f64>("r", 1, &[4.0, 5.0, 6.0]).expect("set r[1]");

    let json = core.serialize().expect("serialize attrs core");
    let value: serde_json::Value = serde_json::from_str(&json).expect("parse attrs core json");

    assert_eq!(value["num_attrs"], 2);
    let attrs = value["attrs"].as_array().expect("attrs array");
    assert_eq!(attrs.len(), 2);

    for item in attrs {
        let payload = &item["payload"];
        assert_eq!(payload["kind"], "vector_list");
        assert_eq!(payload["shape"], serde_json::json!([2, 3]));
        assert!(payload["data"].is_array());
        assert_eq!(payload.as_object().expect("payload object").len(), 3);
    }
}

#[test]
fn phys_obj_serialization_embeds_flat_vector_list_payloads() {
    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 2, 2).expect("allocate r");

    let obj = PhysObj {
        meta: AttrsMeta {
            id: 42,
            label: "serde-flat".to_string(),
            comment: "schema check".to_string(),
        },
        core,
    };

    let json = obj.serialize().expect("serialize phys obj");
    let value: serde_json::Value = serde_json::from_str(&json).expect("parse phys obj json");

    assert_eq!(value["meta"]["id"], 42);
    assert_eq!(value["core"]["num_attrs"], 1);
    assert_eq!(value["core"]["attrs"][0]["payload"]["kind"], "vector_list");
    assert_eq!(value["core"]["attrs"][0]["payload"]["shape"], serde_json::json!([2, 2]));
}

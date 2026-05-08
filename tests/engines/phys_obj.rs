use physics_in_parallel::engines::soa::{AttrId, AttrsCore, AttrsError, AttrsMeta, PhysObj};
use physics_in_parallel::math::tensor::rank_2::vector_list::VectorList;
use std::env;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_tmp_dir(name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    env::temp_dir().join(format!("pip_{name}_{}_{}", std::process::id(), nanos))
}

#[test]
fn attrs_core_allocate_access_and_errors() {
    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 2, 3).unwrap();

    assert_eq!(core.len(), 1);
    assert_eq!(core.n_objects(), Some(3));
    assert_eq!(core.dim_of("r").unwrap(), 2);

    core.set_vector_of::<f64>("r", 1, &[1.0, 2.0]).unwrap();
    assert_eq!(
        core.vector_of::<f64>("r", 1).unwrap(),
        [1.0, 2.0].as_slice()
    );

    let err = core.set_vector_of::<f64>("r", 1, &[1.0]).unwrap_err();
    assert_eq!(
        err,
        AttrsError::WrongVectorLen {
            expected: 2,
            got: 1
        }
    );

    let err = core.vector_of::<f64>("r", 99).unwrap_err();
    assert!(matches!(err, AttrsError::ObjOutOfBounds { .. }));

    let mut charge = VectorList::<f64>::empty(1, 3);
    charge.fill(-1.0);
    core.insert("q", charge).unwrap();
    core.rename("q", "charge").unwrap();
    assert!(core.contains("charge"));
    core.remove("charge").unwrap();
    assert!(!core.contains("charge"));
}

#[test]
fn attrs_core_type_mismatch_and_phys_obj_serialize() {
    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 2, 2).unwrap();

    let err = core.get::<i64>("r").unwrap_err();
    assert!(matches!(err, AttrsError::WrongType { .. }));

    let obj = PhysObj {
        meta: AttrsMeta {
            id: 7 as AttrId,
            label: "unit_test".to_string(),
            comment: "serialize smoke".to_string(),
        },
        core,
    };

    let json = obj.serialize().unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["meta"]["id"], 7);
    assert_eq!(value["meta"]["label"], "unit_test");
    assert_eq!(value["core"]["num_attrs"], 1);
}

#[test]
fn phys_obj_constructors_metadata_and_save_to_json_are_consistent() {
    let empty = PhysObj::empty();
    assert_eq!(empty.meta, AttrsMeta::empty());
    assert!(empty.core.is_empty());
    assert_eq!(empty.core.n_objects(), None);

    let meta = AttrsMeta::new(11, "particles", "test state");
    assert_eq!(meta.id, 11);
    assert_eq!(meta.label, "particles");
    assert_eq!(meta.comment, "test state");

    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 3, 2).unwrap();
    core.allocate::<i64>("species", 1, 2).unwrap();
    core.set_vector_of::<f64>("r", 0, &[1.0, 2.0, 3.0]).unwrap();
    core.set_vector_of::<i64>("species", 1, &[4]).unwrap();

    let labels: Vec<&str> = core.labels().collect();
    assert_eq!(labels.len(), 2);
    assert!(labels.contains(&"r"));
    assert!(labels.contains(&"species"));
    assert_eq!(
        core.type_name_of("r").unwrap(),
        std::any::type_name::<f64>()
    );
    assert_eq!(
        core.type_name_of("species").unwrap(),
        std::any::type_name::<i64>()
    );

    let obj = PhysObj::new(meta, core);
    let out_dir = unique_tmp_dir("phys_obj_save");
    obj.save_to_json(&out_dir, "state.json")
        .expect("PhysObj::save_to_json should create output");

    let raw =
        fs::read_to_string(out_dir.join("state.json")).expect("saved json should be readable");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("saved json should parse");
    assert_eq!(value["meta"]["id"], 11);
    assert_eq!(value["meta"]["label"], "particles");
    assert_eq!(value["core"]["num_attrs"], 2);
    assert_eq!(value["core"]["n_objects"], 2);

    fs::remove_dir_all(out_dir).expect("cleanup saved PhysObj json directory");
}

#[test]
fn attrs_core_rejects_inconsistent_object_count() {
    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 3, 2).unwrap();

    let mismatched = VectorList::<f64>::empty(3, 3);
    let err = core.insert("v", mismatched).unwrap_err();
    assert_eq!(
        err,
        AttrsError::InconsistentObjectCount {
            label: "v".to_string(),
            expected: 2,
            got: 3,
        }
    );
}

#[test]
fn attrs_core_auto_generated_ids_are_stable_and_optional_for_users() {
    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 3, 2).unwrap();
    core.allocate::<f64>("v", 3, 2).unwrap();
    core.allocate::<i64>("species", 1, 2).unwrap();

    let r_id = core.id_of("r").unwrap();
    let v_id = core.id_of("v").unwrap();
    let species_id = core.id_of("species").unwrap();
    assert_ne!(r_id, v_id);
    assert_ne!(v_id, species_id);

    assert_eq!(core.label_of(r_id).unwrap(), "r");
    assert_eq!(core.dim_of_id(v_id).unwrap(), 3);
    assert_eq!(
        core.type_name_of_id(species_id).unwrap(),
        std::any::type_name::<i64>()
    );

    core.get_by_id_mut::<f64>(r_id)
        .unwrap()
        .set_vec(1, &[4.0, 5.0, 6.0]);
    assert_eq!(
        core.get_by_id::<f64>(r_id).unwrap().get_vec(1),
        [4.0, 5.0, 6.0].as_slice()
    );
    assert_eq!(
        core.vector_of::<f64>("r", 1).unwrap(),
        [4.0, 5.0, 6.0].as_slice()
    );
}

#[test]
fn attrs_core_rename_preserves_id_and_remove_invalidates_only_removed_id() {
    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 2, 2).unwrap();
    core.allocate::<f64>("v", 2, 2).unwrap();

    let r_id = core.id_of("r").unwrap();
    let v_id = core.id_of("v").unwrap();

    core.rename("r", "position").unwrap();
    assert_eq!(core.id_of("position").unwrap(), r_id);
    assert!(matches!(
        core.id_of("r").unwrap_err(),
        AttrsError::UnknownLabel { .. }
    ));
    assert_eq!(core.label_of(r_id).unwrap(), "position");

    core.remove("position").unwrap();
    assert!(matches!(
        core.label_of(r_id).unwrap_err(),
        AttrsError::UnknownId { .. }
    ));
    assert_eq!(core.label_of(v_id).unwrap(), "v");

    core.allocate::<f64>("a", 2, 2).unwrap();
    let a_id = core.id_of("a").unwrap();
    assert_ne!(a_id, r_id, "removed attribute id should not be reused");
}

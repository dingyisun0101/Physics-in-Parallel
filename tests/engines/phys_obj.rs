use physics_in_parallel::engines::soa::{AttrId, AttrsCore, AttrsError, AttrsMeta, PhysObj};
use physics_in_parallel::math::tensor::rank_2::vector_list::VectorList;

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

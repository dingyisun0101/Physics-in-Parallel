use physics_in_parallel::math::tensor::rank_2::vector_list::VectorList;
use physics_in_parallel::engines::soa::phys_obj::{PhysObj, PhysObjError};

#[test]
/// Annotation:
/// - Purpose: Executes `phys_obj_empty_defaults_and_core_access` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn phys_obj_empty_defaults_and_core_access() {
    let mut obj = PhysObj::<3>::empty(2);

    assert_eq!(PhysObj::<3>::dim(), 3);
    assert_eq!(obj.len(), 2);
    assert_eq!(obj.num_attrs(), 0);
    assert!(!obj.is_empty());

    assert_eq!(obj.pos_of(0).unwrap(), [0.0, 0.0, 0.0].as_slice());
    assert_eq!(obj.vel_of(0).unwrap(), [0.0, 0.0, 0.0].as_slice());
    assert_eq!(obj.acc_of(0).unwrap(), [0.0, 0.0, 0.0].as_slice());
    assert_eq!(obj.inv_mass_of(0).unwrap(), 1.0);
    assert_eq!(obj.kind_of(0).unwrap(), 0);
    assert!(obj.is_alive(0).unwrap());

    obj.set_pos(0, &[1.0, 2.0, 3.0]).unwrap();
    obj.set_vel(0, &[4.0, 5.0, 6.0]).unwrap();
    obj.set_acc(0, &[7.0, 8.0, 9.0]).unwrap();
    obj.set_inv_mass(0, 2.0).unwrap();
    obj.set_kind(0, 42).unwrap();
    obj.set_alive(0, false).unwrap();

    assert_eq!(obj.pos_of(0).unwrap(), [1.0, 2.0, 3.0].as_slice());
    assert_eq!(obj.vel_of(0).unwrap(), [4.0, 5.0, 6.0].as_slice());
    assert_eq!(obj.acc_of(0).unwrap(), [7.0, 8.0, 9.0].as_slice());
    assert_eq!(obj.inv_mass_of(0).unwrap(), 2.0);
    assert_eq!(obj.kind_of(0).unwrap(), 42);
    assert!(!obj.is_alive(0).unwrap());

    obj.clear_acc();
    assert_eq!(obj.acc_of(0).unwrap(), [0.0, 0.0, 0.0].as_slice());
}

#[test]
/// Annotation:
/// - Purpose: Executes `phys_obj_attr_registration_lookup_and_errors` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn phys_obj_attr_registration_lookup_and_errors() {
    let mut obj = PhysObj::<2>::empty(3);

    let charge = obj
        .register_attr("charge", -1.0, Some("electrostatic charge".to_string()))
        .unwrap();
    assert_eq!(charge, 0);
    assert_eq!(obj.attr_id_of("charge"), Some(charge));
    assert_eq!(obj.attr_schema().len(), 1);
    assert_eq!(obj.attr_schema().meta(charge).unwrap().default, -1.0);
    assert_eq!(obj.attr_schema().meta(charge).unwrap().name, "charge");

    // Defaults are broadcast to all existing objects.
    for i in 0..obj.len() {
        assert_eq!(obj.attr(i, charge).unwrap(), -1.0);
    }

    obj.set_attr(1, charge, 2.5).unwrap();
    obj.set_attr_by_name(2, "charge", -3.0).unwrap();
    assert_eq!(obj.attr(1, charge).unwrap(), 2.5);
    assert_eq!(obj.attr_by_name(2, "charge").unwrap(), -3.0);

    let dup = obj
        .register_attr("charge", 0.0, None)
        .expect_err("duplicate attr name must fail");
    assert_eq!(
        dup,
        PhysObjError::DuplicateAttrName {
            name: "charge".to_string()
        }
    );

    assert_eq!(
        obj.attr(99, charge),
        Err(PhysObjError::InvalidObjId {
            obj: 99,
            n: obj.len()
        })
    );

    assert_eq!(
        obj.attr(0, 99),
        Err(PhysObjError::InvalidAttrId {
            attr: 99,
            n_attrs: 1
        })
    );

    assert_eq!(
        obj.attr_by_name(0, "unknown"),
        Err(PhysObjError::UnknownAttrName {
            name: "unknown".to_string()
        })
    );
}

#[test]
/// Annotation:
/// - Purpose: Executes `phys_obj_push_object_propagates_attr_defaults_and_preserves_existing_data` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn phys_obj_push_object_propagates_attr_defaults_and_preserves_existing_data() {
    let mut obj = PhysObj::<2>::empty(1);

    obj.set_pos(0, &[10.0, 20.0]).unwrap();
    obj.set_vel(0, &[1.0, 2.0]).unwrap();
    obj.set_acc(0, &[3.0, 4.0]).unwrap();
    obj.set_inv_mass(0, 5.0).unwrap();
    obj.set_kind(0, 7).unwrap();

    let temperature = obj.register_attr("temperature", 273.15, None).unwrap();
    obj.set_attr(0, temperature, 350.0).unwrap();

    let new_id = obj.push_object();
    assert_eq!(new_id, 1);
    assert_eq!(obj.len(), 2);

    // Existing object remains unchanged.
    assert_eq!(obj.pos_of(0).unwrap(), [10.0, 20.0].as_slice());
    assert_eq!(obj.vel_of(0).unwrap(), [1.0, 2.0].as_slice());
    assert_eq!(obj.acc_of(0).unwrap(), [3.0, 4.0].as_slice());
    assert_eq!(obj.inv_mass_of(0).unwrap(), 5.0);
    assert_eq!(obj.kind_of(0).unwrap(), 7);
    assert_eq!(obj.attr(0, temperature).unwrap(), 350.0);

    // Newly appended object gets default core + attribute values.
    assert_eq!(obj.pos_of(new_id).unwrap(), [0.0, 0.0].as_slice());
    assert_eq!(obj.vel_of(new_id).unwrap(), [0.0, 0.0].as_slice());
    assert_eq!(obj.acc_of(new_id).unwrap(), [0.0, 0.0].as_slice());
    assert_eq!(obj.inv_mass_of(new_id).unwrap(), 1.0);
    assert_eq!(obj.kind_of(new_id).unwrap(), 0);
    assert!(obj.is_alive(new_id).unwrap());
    assert_eq!(obj.attr(new_id, temperature).unwrap(), 273.15);
}

#[test]
/// Annotation:
/// - Purpose: Executes `phys_obj_kill_object_and_invalid_id_paths` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn phys_obj_kill_object_and_invalid_id_paths() {
    let mut obj = PhysObj::<2>::empty(2);

    assert!(obj.is_alive(1).unwrap());
    obj.kill_object(1).unwrap();
    assert!(!obj.is_alive(1).unwrap());

    assert_eq!(
        obj.kill_object(9),
        Err(PhysObjError::InvalidObjId { obj: 9, n: 2 })
    );

    assert_eq!(
        obj.set_inv_mass(0, 0.0),
        Err(PhysObjError::InvalidCoreValue {
            field: "inv_mass",
            value: 0.0
        })
    );
}

#[test]
/// Annotation:
/// - Purpose: Executes `phys_obj_from_core_accepts_valid_and_rejects_invalid_inputs` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn phys_obj_from_core_accepts_valid_and_rejects_invalid_inputs() {
    let mut pos = VectorList::<f64>::empty(2, 2);
    let mut vel = VectorList::<f64>::empty(2, 2);
    let mut acc = VectorList::<f64>::empty(2, 2);

    pos.set_vector_from_slice(0, &[1.0, 2.0]);
    pos.set_vector_from_slice(1, &[3.0, 4.0]);
    vel.set_vector_from_slice(0, &[5.0, 6.0]);
    vel.set_vector_from_slice(1, &[7.0, 8.0]);
    acc.set_vector_from_slice(0, &[9.0, 10.0]);
    acc.set_vector_from_slice(1, &[11.0, 12.0]);

    let good = PhysObj::<2>::from_core(
        pos.clone(),
        vel.clone(),
        acc.clone(),
        vec![1.0, 2.0],
        vec![true, false],
        vec![4, 5],
    )
    .unwrap();

    assert_eq!(good.len(), 2);
    assert_eq!(good.kind_of(0).unwrap(), 4);
    assert_eq!(good.kind_of(1).unwrap(), 5);
    assert_eq!(good.pos_of(1).unwrap(), [3.0, 4.0].as_slice());

    let bad_shape = PhysObj::<2>::from_core(
        VectorList::<f64>::empty(2, 1), // wrong n
        vel.clone(),
        acc.clone(),
        vec![1.0, 2.0],
        vec![true, true],
        vec![0, 0],
    )
    .expect_err("shape mismatch should fail");
    assert_eq!(bad_shape, PhysObjError::WrongVectorLen { expected: 4, got: 2 });

    let bad_mass = PhysObj::<2>::from_core(
        pos,
        vel,
        acc,
        vec![1.0, -0.1],
        vec![true, true],
        vec![0, 0],
    )
    .expect_err("negative inv_mass should fail");
    assert_eq!(
        bad_mass,
        PhysObjError::InvalidCoreValue {
            field: "inv_mass",
            value: -0.1
        }
    );
}

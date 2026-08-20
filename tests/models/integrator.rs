use physics_in_parallel::prelude::advanced::PhysObjAdvanced;
use physics_in_parallel::prelude::basic::*;
use physics_in_parallel::prelude::models::*;

#[test]
fn euler_updates_v_then_r() {
    let mut obj = create_template(1, 1).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_R, 0, &[0.0]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 0, &[1.0]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_A, 0, &[2.0]).unwrap();

    let mut explicit = ExplicitEuler;
    explicit.apply(&mut obj, 0.5).unwrap();
    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_V, 0).unwrap(),
        [2.0].as_slice()
    );
    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_R, 0).unwrap(),
        [0.5].as_slice()
    );

    let mut obj2 = create_template(1, 1).unwrap();
    obj2.set_attribute_vector::<f64>(ATTR_R, 0, &[0.0]).unwrap();
    obj2.set_attribute_vector::<f64>(ATTR_V, 0, &[1.0]).unwrap();
    obj2.set_attribute_vector::<f64>(ATTR_A, 0, &[2.0]).unwrap();

    let mut semi = SemiImplicitEuler;
    semi.apply(&mut obj2, 0.5).unwrap();
    assert_eq!(
        obj2.attribute_vector::<f64>(ATTR_V, 0).unwrap(),
        [2.0].as_slice()
    );
    assert_eq!(
        obj2.attribute_vector::<f64>(ATTR_R, 0).unwrap(),
        [1.0].as_slice()
    );
}

#[test]
fn integrator_respects_alive_mask_and_validates_dt() {
    let mut obj = create_template(1, 1).unwrap();
    set_alive(&mut obj, 0, false).unwrap();

    obj.set_attribute_vector::<f64>(ATTR_R, 0, &[3.0]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 0, &[4.0]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_A, 0, &[5.0]).unwrap();

    let mut explicit = ExplicitEuler;
    explicit.apply(&mut obj, 0.5).unwrap();

    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_R, 0).unwrap(),
        [3.0].as_slice()
    );
    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_V, 0).unwrap(),
        [4.0].as_slice()
    );

    let err = explicit.apply(&mut obj, 0.0).unwrap_err();
    assert_eq!(err, IntegratorError::InvalidDt { dt: 0.0 });
}

#[test]
fn integrator_respects_rigid_mask_and_does_not_clear_acceleration() {
    let mut obj = create_template(1, 2).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_R, 0, &[0.0]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 0, &[1.0]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_A, 0, &[2.0]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_R, 1, &[10.0]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 1, &[3.0]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_A, 1, &[4.0]).unwrap();
    set_rigid(&mut obj, 1, true).unwrap();

    let mut semi = SemiImplicitEuler;
    semi.apply(&mut obj, 0.5).unwrap();

    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_R, 0).unwrap(),
        [1.0].as_slice()
    );
    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_V, 0).unwrap(),
        [2.0].as_slice()
    );
    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_R, 1).unwrap(),
        [10.0].as_slice()
    );
    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_V, 1).unwrap(),
        [3.0].as_slice()
    );
    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_A, 1).unwrap(),
        [4.0].as_slice()
    );
}

#[test]
fn integrator_reports_shape_errors() {
    let mut bad_accel = create_template(1, 2).unwrap();
    bad_accel.attributes_mut().remove(ATTR_A).unwrap();
    bad_accel
        .attributes_mut()
        .insert(ATTR_A, VectorList::<f64>::empty(2, 2))
        .unwrap();

    let mut explicit = ExplicitEuler;
    assert_eq!(
        explicit.apply(&mut bad_accel, 0.1).unwrap_err(),
        IntegratorError::InvalidAttrShape {
            label: ATTR_A,
            expected_dim: 1,
            got_dim: 2,
        }
    );

    let mut bad_rigid = create_template(1, 2).unwrap();
    bad_rigid.attributes_mut().remove(ATTR_RIGID).unwrap();
    bad_rigid
        .attributes_mut()
        .insert(ATTR_RIGID, VectorList::<u8>::empty(2, 2))
        .unwrap();

    assert_eq!(
        explicit.apply(&mut bad_rigid, 0.1).unwrap_err(),
        IntegratorError::InvalidAttrShape {
            label: ATTR_RIGID,
            expected_dim: 1,
            got_dim: 2,
        }
    );
}

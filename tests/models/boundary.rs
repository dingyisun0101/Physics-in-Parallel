use physics_in_parallel::prelude::basic::*;
use physics_in_parallel::prelude::models::*;

#[test]
fn periodic_and_clamp_update_positions() {
    let mut obj = create_template(1, 2).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_R, 0, &[1.3]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_R, 1, &[-0.2]).unwrap();

    PeriodicBox::new(&[0.0], &[1.0])
        .unwrap()
        .apply_to_particles(&mut obj)
        .unwrap();
    assert!((obj.attribute_vector::<f64>(ATTR_R, 0).unwrap()[0] - 0.3).abs() < 1e-12);
    assert!((obj.attribute_vector::<f64>(ATTR_R, 1).unwrap()[0] - 0.8).abs() < 1e-12);

    obj.set_attribute_vector::<f64>(ATTR_R, 0, &[1.3]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_R, 1, &[-0.2]).unwrap();

    ClampBox::new(&[0.0], &[1.0])
        .unwrap()
        .apply_to_particles(&mut obj)
        .unwrap();
    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_R, 0).unwrap(),
        [1.0].as_slice()
    );
    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_R, 1).unwrap(),
        [0.0].as_slice()
    );
}

#[test]
fn reflect_flips_velocity_and_respects_alive_mask() {
    let mut obj = create_template(1, 2).unwrap();

    obj.set_attribute_vector::<f64>(ATTR_R, 0, &[1.2]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 0, &[2.0]).unwrap();
    set_alive(&mut obj, 0, true).unwrap();

    obj.set_attribute_vector::<f64>(ATTR_R, 1, &[1.2]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 1, &[3.0]).unwrap();
    set_alive(&mut obj, 1, false).unwrap();

    ReflectBox::new(&[0.0], &[1.0])
        .unwrap()
        .apply_to_particles(&mut obj)
        .unwrap();

    assert!((obj.attribute_vector::<f64>(ATTR_R, 0).unwrap()[0] - 0.8).abs() < 1e-12);
    assert!((obj.attribute_vector::<f64>(ATTR_V, 0).unwrap()[0] + 2.0).abs() < 1e-12);

    assert!((obj.attribute_vector::<f64>(ATTR_R, 1).unwrap()[0] - 1.2).abs() < 1e-12);
    assert!((obj.attribute_vector::<f64>(ATTR_V, 1).unwrap()[0] - 3.0).abs() < 1e-12);
}

#[test]
fn reflect_handles_large_overshoot_per_axis() {
    let mut obj = create_template(2, 2).unwrap();

    obj.set_attribute_vector::<f64>(ATTR_R, 0, &[1.2, -0.25])
        .unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 0, &[2.0, -3.0])
        .unwrap();

    obj.set_attribute_vector::<f64>(ATTR_R, 1, &[2.2, -1.2])
        .unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 1, &[5.0, 7.0])
        .unwrap();

    ReflectBox::new(&[0.0, 0.0], &[1.0, 1.0])
        .unwrap()
        .apply_to_particles(&mut obj)
        .unwrap();

    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_R, 0).unwrap(),
        [0.8, 0.25].as_slice()
    );
    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_V, 0).unwrap(),
        [-2.0, 3.0].as_slice()
    );

    assert!((obj.attribute_vector::<f64>(ATTR_R, 1).unwrap()[0] - 0.2).abs() < 1e-12);
    assert!((obj.attribute_vector::<f64>(ATTR_R, 1).unwrap()[1] - 0.8).abs() < 1e-12);
    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_V, 1).unwrap(),
        [5.0, 7.0].as_slice()
    );
}

#[test]
fn boundary_skips_rigid_particles() {
    let mut obj = create_template(1, 2).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_R, 0, &[1.2]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_R, 1, &[1.2]).unwrap();
    set_rigid(&mut obj, 0, true).unwrap();
    set_rigid(&mut obj, 1, false).unwrap();

    PeriodicBox::new(&[0.0], &[1.0])
        .unwrap()
        .apply_to_particles(&mut obj)
        .unwrap();

    assert_eq!(
        obj.attribute_vector::<f64>(ATTR_R, 0).unwrap(),
        [1.2].as_slice()
    );
    assert!((obj.attribute_vector::<f64>(ATTR_R, 1).unwrap()[0] - 0.2).abs() < 1e-12);
}

#[test]
fn boundary_rejects_dimension_mismatch_without_panicking() {
    let mut obj = create_template(2, 1).unwrap();
    let err = PeriodicBox::new(&[0.0], &[1.0])
        .unwrap()
        .apply_to_particles(&mut obj)
        .unwrap_err();

    assert_eq!(
        err,
        ParticleBoundaryError::Boundary(BoundaryError::InvalidVectorDimension {
            label: "bounds",
            expected: 2,
            got: 1,
        })
    );
}

use physics_in_parallel::models::particles::attrs::{ATTR_ALIVE, ATTR_R, ATTR_V};
use physics_in_parallel::models::particles::boundary::{
    Boundary, ClampBox, PeriodicBox, ReflectBox,
};
use physics_in_parallel::models::particles::create_state::create_template;

#[test]
fn periodic_and_clamp_update_positions() {
    let mut obj = create_template(1, 2).unwrap();
    obj.core.set_vector_of::<f64>(ATTR_R, 0, &[1.3]).unwrap();
    obj.core.set_vector_of::<f64>(ATTR_R, 1, &[-0.2]).unwrap();

    PeriodicBox::new(&[0.0], &[1.0])
        .unwrap()
        .apply(&mut obj)
        .unwrap();
    assert!((obj.core.vector_of::<f64>(ATTR_R, 0).unwrap()[0] - 0.3).abs() < 1e-12);
    assert!((obj.core.vector_of::<f64>(ATTR_R, 1).unwrap()[0] - 0.8).abs() < 1e-12);

    obj.core.set_vector_of::<f64>(ATTR_R, 0, &[1.3]).unwrap();
    obj.core.set_vector_of::<f64>(ATTR_R, 1, &[-0.2]).unwrap();

    ClampBox::new(&[0.0], &[1.0])
        .unwrap()
        .apply(&mut obj)
        .unwrap();
    assert_eq!(
        obj.core.vector_of::<f64>(ATTR_R, 0).unwrap(),
        [1.0].as_slice()
    );
    assert_eq!(
        obj.core.vector_of::<f64>(ATTR_R, 1).unwrap(),
        [0.0].as_slice()
    );
}

#[test]
fn reflect_flips_velocity_and_respects_alive_mask() {
    let mut obj = create_template(1, 2).unwrap();
    obj.core.allocate::<f64>(ATTR_ALIVE, 1, 2).unwrap();

    obj.core.set_vector_of::<f64>(ATTR_R, 0, &[1.2]).unwrap();
    obj.core.set_vector_of::<f64>(ATTR_V, 0, &[2.0]).unwrap();
    obj.core
        .set_vector_of::<f64>(ATTR_ALIVE, 0, &[1.0])
        .unwrap();

    obj.core.set_vector_of::<f64>(ATTR_R, 1, &[1.2]).unwrap();
    obj.core.set_vector_of::<f64>(ATTR_V, 1, &[3.0]).unwrap();
    obj.core
        .set_vector_of::<f64>(ATTR_ALIVE, 1, &[0.0])
        .unwrap();

    ReflectBox::new(&[0.0], &[1.0])
        .unwrap()
        .apply(&mut obj)
        .unwrap();

    assert!((obj.core.vector_of::<f64>(ATTR_R, 0).unwrap()[0] - 0.8).abs() < 1e-12);
    assert!((obj.core.vector_of::<f64>(ATTR_V, 0).unwrap()[0] + 2.0).abs() < 1e-12);

    assert!((obj.core.vector_of::<f64>(ATTR_R, 1).unwrap()[0] - 1.2).abs() < 1e-12);
    assert!((obj.core.vector_of::<f64>(ATTR_V, 1).unwrap()[0] - 3.0).abs() < 1e-12);
}

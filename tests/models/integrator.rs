use physics_in_parallel::models::particles::attrs::{ATTR_A, ATTR_ALIVE, ATTR_R, ATTR_V};
use physics_in_parallel::models::particles::create_state::create_template;
use physics_in_parallel::models::particles::integrator::{
    ExplicitEuler, Integrator, IntegratorError, SemiImplicitEuler,
};

#[test]
fn euler_updates_v_then_r() {
    let mut obj = create_template(1, 1).unwrap();
    obj.core.set_vector_of::<f64>(ATTR_R, 0, &[0.0]).unwrap();
    obj.core.set_vector_of::<f64>(ATTR_V, 0, &[1.0]).unwrap();
    obj.core.set_vector_of::<f64>(ATTR_A, 0, &[2.0]).unwrap();

    let mut explicit = ExplicitEuler;
    explicit.apply(&mut obj, 0.5).unwrap();
    assert_eq!(
        obj.core.vector_of::<f64>(ATTR_V, 0).unwrap(),
        [2.0].as_slice()
    );
    assert_eq!(
        obj.core.vector_of::<f64>(ATTR_R, 0).unwrap(),
        [0.5].as_slice()
    );

    let mut obj2 = create_template(1, 1).unwrap();
    obj2.core.set_vector_of::<f64>(ATTR_R, 0, &[0.0]).unwrap();
    obj2.core.set_vector_of::<f64>(ATTR_V, 0, &[1.0]).unwrap();
    obj2.core.set_vector_of::<f64>(ATTR_A, 0, &[2.0]).unwrap();

    let mut semi = SemiImplicitEuler;
    semi.apply(&mut obj2, 0.5).unwrap();
    assert_eq!(
        obj2.core.vector_of::<f64>(ATTR_V, 0).unwrap(),
        [2.0].as_slice()
    );
    assert_eq!(
        obj2.core.vector_of::<f64>(ATTR_R, 0).unwrap(),
        [1.0].as_slice()
    );
}

#[test]
fn integrator_respects_alive_mask_and_validates_dt() {
    let mut obj = create_template(1, 1).unwrap();
    obj.core.allocate::<f64>(ATTR_ALIVE, 1, 1).unwrap();
    obj.core
        .set_vector_of::<f64>(ATTR_ALIVE, 0, &[0.0])
        .unwrap();

    obj.core.set_vector_of::<f64>(ATTR_R, 0, &[3.0]).unwrap();
    obj.core.set_vector_of::<f64>(ATTR_V, 0, &[4.0]).unwrap();
    obj.core.set_vector_of::<f64>(ATTR_A, 0, &[5.0]).unwrap();

    let mut explicit = ExplicitEuler;
    explicit.apply(&mut obj, 0.5).unwrap();

    assert_eq!(
        obj.core.vector_of::<f64>(ATTR_R, 0).unwrap(),
        [3.0].as_slice()
    );
    assert_eq!(
        obj.core.vector_of::<f64>(ATTR_V, 0).unwrap(),
        [4.0].as_slice()
    );

    let err = explicit.apply(&mut obj, 0.0).unwrap_err();
    assert_eq!(err, IntegratorError::InvalidDt { dt: 0.0 });
}

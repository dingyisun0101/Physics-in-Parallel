use physics_in_parallel::models::particles::create_state::{
    create_template, randomize_r, randomize_v, MassiveParticlesError, RandPosMethod, RandVelMethod,
    ATTR_M, ATTR_M_INV, ATTR_R, ATTR_V,
};

#[test]
fn create_template_shapes_and_defaults() {
    let obj = create_template(3, 4).unwrap();

    let r = obj.core.get::<f64>(ATTR_R).unwrap();
    let v = obj.core.get::<f64>(ATTR_V).unwrap();
    let m = obj.core.get::<f64>(ATTR_M).unwrap();
    let m_inv = obj.core.get::<f64>(ATTR_M_INV).unwrap();

    assert_eq!(r.dim(), 3);
    assert_eq!(r.num_vectors(), 4);
    assert_eq!(v.dim(), 3);
    assert_eq!(v.num_vectors(), 4);
    assert_eq!(m.dim(), 1);
    assert_eq!(m_inv.dim(), 1);

    for i in 0..4 {
        assert_eq!(m.get(i as isize, 0), 1.0);
        assert_eq!(m_inv.get(i as isize, 0), 1.0);
    }
}

#[test]
fn randomize_r_uniform_stays_inside_box() {
    let mut obj = create_template(2, 256).unwrap();

    randomize_r(
        &mut obj,
        RandPosMethod::Uniform {
            box_size: &[2.0, 4.0],
        },
    )
    .unwrap();

    let r = obj.core.get::<f64>(ATTR_R).unwrap();
    for i in 0..r.num_vectors() {
        let row = r.get_vector(i as isize);
        assert!(row[0] >= -1.0 && row[0] <= 1.0);
        assert!(row[1] >= -2.0 && row[1] <= 2.0);
    }
}

#[test]
fn randomize_v_invalid_and_mass_inv_validation() {
    let mut obj = create_template(2, 3).unwrap();

    let err = randomize_v(
        &mut obj,
        RandVelMethod::Uniform {
            low: 1.0,
            high: 1.0,
        },
    )
    .unwrap_err();
    assert!(matches!(err, MassiveParticlesError::Distribution { .. }));

    obj.core
        .set_vector_of::<f64>(ATTR_M_INV, 1, &[-1.0])
        .unwrap();

    let err = randomize_v(&mut obj, RandVelMethod::MaxwellBoltzmann { tau: 1.0 }).unwrap_err();
    assert!(matches!(
        err,
        MassiveParticlesError::InvalidMassInv {
            index: 1,
            value: -1.0
        }
    ));
}

use physics_in_parallel::prelude::basic::*;
use physics_in_parallel::prelude::models::*;

#[test]
fn create_template_shapes_and_defaults() {
    let obj = create_template(3, 4).unwrap();

    let r = obj.attribute::<f64>(ATTR_R).unwrap();
    let v = obj.attribute::<f64>(ATTR_V).unwrap();
    let m = obj.attribute::<f64>(ATTR_M).unwrap();
    let m_inv = obj.attribute::<f64>(ATTR_M_INV).unwrap();
    let rigid = obj.attribute::<u8>(ATTR_RIGID).unwrap();

    assert_eq!(r.dim(), 3);
    assert_eq!(r.num_vectors(), 4);
    assert_eq!(v.dim(), 3);
    assert_eq!(v.num_vectors(), 4);
    assert_eq!(m.dim(), 1);
    assert_eq!(m_inv.dim(), 1);
    assert_eq!(rigid.dim(), 1);
    assert_eq!(obj.label(), "particles");

    for i in 0..4 {
        assert_eq!(m.get(i as isize, 0), 1.0);
        assert_eq!(m_inv.get(i as isize, 0), 1.0);
        assert!(is_alive(&obj, i).unwrap());
        assert!(!is_rigid(&obj, i).unwrap());
    }

    let mut obj = obj;
    set_alive(&mut obj, 2, false).unwrap();
    assert!(!is_alive(&obj, 2).unwrap());
    set_rigid(&mut obj, 1, true).unwrap();
    assert!(is_rigid(&obj, 1).unwrap());
}

#[test]
fn create_template_rejects_zero_shape_inputs() {
    assert_eq!(
        create_template(0, 4).unwrap_err(),
        MassiveParticlesError::InvalidDimension { dim: 0 }
    );
    assert_eq!(
        create_template(3, 0).unwrap_err(),
        MassiveParticlesError::InvalidParticleCount { n: 0 }
    );
}

#[test]
fn randomize_r_uniform_stays_inside_box() {
    let mut obj = create_template(2, 256).unwrap();

    randomize_r(
        &mut obj,
        VectorSamplingMethod::UniformCentered {
            box_size: &[2.0, 4.0],
        },
        RngConfig::default(),
    )
    .unwrap();

    let r = obj.attribute::<f64>(ATTR_R).unwrap();
    for i in 0..r.num_vectors() {
        let row = r.vector(i as isize);
        assert!(row[0] >= -1.0 && row[0] <= 1.0);
        assert!(row[1] >= -2.0 && row[1] <= 2.0);
    }
}

#[test]
fn randomize_r_rejects_invalid_position_parameters() {
    let mut obj = create_template(2, 4).unwrap();

    let err = randomize_r(
        &mut obj,
        VectorSamplingMethod::UniformCentered { box_size: &[1.0] },
        RngConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(err, MassiveParticlesError::Distribution { .. }));

    let err = randomize_r(
        &mut obj,
        VectorSamplingMethod::JitteredLattice {
            spacings: &[1.0, f64::NAN],
            sigmas: &[0.0, 0.0],
        },
        RngConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(err, MassiveParticlesError::Distribution { .. }));
}

#[test]
fn randomize_v_invalid_and_mass_inv_validation() {
    let mut obj = create_template(2, 3).unwrap();

    let err = randomize_v(
        &mut obj,
        VelocitySamplingMethod::Uniform {
            low: 1.0,
            high: 1.0,
        },
        RngConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(err, MassiveParticlesError::Distribution { .. }));

    obj.set_attribute_vector::<f64>(ATTR_M_INV, 1, &[-1.0])
        .unwrap();

    let err = randomize_v(
        &mut obj,
        VelocitySamplingMethod::MaxwellBoltzmann { tau: 1.0 },
        RngConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        MassiveParticlesError::InvalidMassInv {
            index: 1,
            value: -1.0
        }
    ));
}

#[test]
fn randomize_v_gaussian_per_axis_validates_parameters() {
    let mut obj = create_template(2, 3).unwrap();

    let err = randomize_v(
        &mut obj,
        VelocitySamplingMethod::GaussianPerAxis {
            mean: &[0.0],
            std: &[1.0, 1.0],
        },
        RngConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(err, MassiveParticlesError::Distribution { .. }));

    let err = randomize_v(
        &mut obj,
        VelocitySamplingMethod::GaussianPerAxis {
            mean: &[0.0, 0.0],
            std: &[1.0, -1.0],
        },
        RngConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(err, MassiveParticlesError::Distribution { .. }));
}

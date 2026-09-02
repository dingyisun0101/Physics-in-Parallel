use physics_in_parallel::prelude::advanced::PhysObjAdvanced;
use physics_in_parallel::prelude::basic::{RngConfig, RngConfigError, VectorList};
use physics_in_parallel::prelude::models::*;

fn rng(seed: u64) -> RngConfig {
    RngConfig::new(Some(seed), None)
}

#[test]
fn langevin_deterministic_for_same_seed_and_state() {
    let mut a = create_template(2, 2).unwrap();
    let mut b = a.clone();

    a.set_attribute_vector::<f64>(ATTR_V, 0, &[1.0, -1.0])
        .unwrap();
    a.set_attribute_vector::<f64>(ATTR_V, 1, &[0.5, 2.0])
        .unwrap();
    b.set_attribute_vector::<f64>(ATTR_V, 0, &[1.0, -1.0])
        .unwrap();
    b.set_attribute_vector::<f64>(ATTR_V, 1, &[0.5, 2.0])
        .unwrap();

    let mut ta = LangevinThermostat::new(1.0, 0.7, rng(42), ParticleSelection::AliveOnly).unwrap();
    let mut tb = LangevinThermostat::new(1.0, 0.7, rng(42), ParticleSelection::AliveOnly).unwrap();

    ta.apply(&mut a, 0.05).unwrap();
    tb.apply(&mut b, 0.05).unwrap();
    assert_eq!(ta.step_counter(), 1);
    assert_eq!(tb.step_counter(), 1);

    for i in 0..2 {
        assert_eq!(
            a.attribute_vector::<f64>(ATTR_V, i).unwrap(),
            b.attribute_vector::<f64>(ATTR_V, i).unwrap()
        );
    }
}

#[test]
fn langevin_restores_the_exact_next_random_step() {
    let mut original_particles = create_template(2, 2).unwrap();
    original_particles
        .set_attribute_vector::<f64>(ATTR_V, 0, &[1.0, -1.0])
        .unwrap();
    original_particles
        .set_attribute_vector::<f64>(ATTR_V, 1, &[0.5, 2.0])
        .unwrap();

    let mut original = LangevinThermostat::new(1.5, 0.4, rng(77), ParticleSelection::All).unwrap();
    original.apply(&mut original_particles, 0.05).unwrap();

    let mut restored_particles = original_particles.clone();
    let mut restored = LangevinThermostat::from_state(
        original.tau_target(),
        original.gamma(),
        original.rng_config(),
        original.step_counter(),
        original.selection(),
    )
    .unwrap();

    original.apply(&mut original_particles, 0.05).unwrap();
    restored.apply(&mut restored_particles, 0.05).unwrap();

    assert_eq!(restored.step_counter(), original.step_counter());
    for particle in 0..2 {
        assert_eq!(
            restored_particles
                .attribute_vector::<f64>(ATTR_V, particle)
                .unwrap(),
            original_particles
                .attribute_vector::<f64>(ATTR_V, particle)
                .unwrap()
        );
    }
}

#[test]
fn langevin_restore_requires_resolved_rng_state() {
    assert!(matches!(
        LangevinThermostat::from_state(
            1.0,
            0.5,
            RngConfig::default(),
            4,
            ParticleSelection::AliveOnly,
        ),
        Err(ThermostatError::RngConfig(
            RngConfigError::MissingSeed { .. }
        ))
    ));
}

#[test]
fn langevin_zero_gamma_keeps_velocity_unchanged() {
    let mut obj = create_template(1, 1).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 0, &[3.5]).unwrap();

    let mut t = LangevinThermostat::new(2.0, 0.0, rng(9), ParticleSelection::AliveOnly).unwrap();
    t.apply(&mut obj, 0.1).unwrap();

    assert!((obj.attribute_vector::<f64>(ATTR_V, 0).unwrap()[0] - 3.5).abs() < 1e-12);
}

#[test]
fn langevin_rejects_invalid_dt() {
    let mut obj = create_template(1, 1).unwrap();
    let mut t = LangevinThermostat::new(1.0, 0.5, rng(1), ParticleSelection::AliveOnly).unwrap();

    assert_eq!(
        t.apply(&mut obj, 0.0),
        Err(ThermostatError::InvalidDt { dt: 0.0 })
    );
}

#[test]
fn langevin_rejects_invalid_constructor_parameters() {
    match LangevinThermostat::new(f64::NAN, 0.1, rng(1), ParticleSelection::AliveOnly).unwrap_err()
    {
        ThermostatError::InvalidParam { field, value } => {
            assert_eq!(field, "tau_target");
            assert!(value.is_nan());
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(
        LangevinThermostat::new(1.0, -0.1, rng(1), ParticleSelection::AliveOnly).unwrap_err(),
        ThermostatError::InvalidParam {
            field: "gamma",
            value: -0.1,
        }
    );
}

#[test]
fn langevin_respects_alive_selection_and_rigid_mask() {
    let mut alive_only = create_template(1, 3).unwrap();
    let mut all = create_template(1, 3).unwrap();
    for obj in [&mut alive_only, &mut all] {
        obj.set_attribute_vector::<f64>(ATTR_V, 0, &[1.0]).unwrap();
        obj.set_attribute_vector::<f64>(ATTR_V, 1, &[2.0]).unwrap();
        obj.set_attribute_vector::<f64>(ATTR_V, 2, &[3.0]).unwrap();
        set_alive(obj, 1, false).unwrap();
        set_rigid(obj, 2, true).unwrap();
    }

    let mut t_alive =
        LangevinThermostat::new(1.0, 0.5, rng(7), ParticleSelection::AliveOnly).unwrap();
    t_alive.apply(&mut alive_only, 0.2).unwrap();

    let mut t_all = LangevinThermostat::new(1.0, 0.5, rng(7), ParticleSelection::All).unwrap();
    t_all.apply(&mut all, 0.2).unwrap();

    assert_ne!(
        alive_only.attribute_vector::<f64>(ATTR_V, 0).unwrap()[0],
        1.0
    );
    assert_eq!(
        alive_only.attribute_vector::<f64>(ATTR_V, 1).unwrap()[0],
        2.0
    );
    assert_eq!(
        alive_only.attribute_vector::<f64>(ATTR_V, 2).unwrap()[0],
        3.0
    );

    assert_ne!(all.attribute_vector::<f64>(ATTR_V, 1).unwrap()[0], 2.0);
    assert_eq!(all.attribute_vector::<f64>(ATTR_V, 2).unwrap()[0], 3.0);
}

#[test]
fn langevin_reports_invalid_inverse_mass_only_for_included_particles() {
    let mut obj = create_template(1, 2).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_M_INV, 0, &[0.0])
        .unwrap();

    let mut t = LangevinThermostat::new(1.0, 0.5, rng(1), ParticleSelection::AliveOnly).unwrap();
    assert_eq!(
        t.apply(&mut obj, 0.1).unwrap_err(),
        ThermostatError::InvalidParam {
            field: ATTR_M_INV,
            value: 0.0,
        }
    );

    set_alive(&mut obj, 0, false).unwrap();
    t.apply(&mut obj, 0.1).unwrap();
}

#[test]
fn langevin_reports_shape_errors() {
    let mut bad_m_inv = create_template(1, 2).unwrap();
    bad_m_inv.attributes_mut().remove(ATTR_M_INV).unwrap();
    bad_m_inv
        .attributes_mut()
        .insert(ATTR_M_INV, VectorList::<f64>::empty(2, 2))
        .unwrap();
    let mut t = LangevinThermostat::new(1.0, 0.5, rng(1), ParticleSelection::AliveOnly).unwrap();
    assert_eq!(
        t.apply(&mut bad_m_inv, 0.1).unwrap_err(),
        ThermostatError::InvalidAttrShape {
            label: ATTR_M_INV,
            expected_dim: 1,
            got_dim: 2,
        }
    );

    let mut bad_alive = create_template(1, 2).unwrap();
    bad_alive.attributes_mut().remove(ATTR_ALIVE).unwrap();
    bad_alive
        .attributes_mut()
        .insert(ATTR_ALIVE, VectorList::<u8>::empty(2, 2))
        .unwrap();
    assert_eq!(
        t.apply(&mut bad_alive, 0.1).unwrap_err(),
        ThermostatError::InvalidAttrShape {
            label: ATTR_ALIVE,
            expected_dim: 1,
            got_dim: 2,
        }
    );
}

use physics_in_parallel::engines::soa::{
    phys_obj::PhysObj,
    thermostat::{langevin::LangevinThermostat, Thermostat, ThermostatError},
};

#[test]
/// Annotation:
/// - Purpose: Executes `langevin_is_deterministic_given_same_seed_and_state` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn langevin_is_deterministic_given_same_seed_and_state() {
    let mut a = PhysObj::<2>::empty(2);
    let mut b = a.clone();

    let mut ta = LangevinThermostat::new(1.0, 0.7, 42, false).unwrap();
    let mut tb = LangevinThermostat::new(1.0, 0.7, 42, false).unwrap();

    ta.apply(&mut a, 0.05).unwrap();
    tb.apply(&mut b, 0.05).unwrap();

    for i in 0..2 {
        assert_eq!(a.vel_of(i).unwrap(), b.vel_of(i).unwrap());
    }
}

#[test]
/// Annotation:
/// - Purpose: Executes `langevin_with_zero_gamma_leaves_velocity_unchanged` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn langevin_with_zero_gamma_leaves_velocity_unchanged() {
    let mut obj = PhysObj::<1>::empty(1);
    obj.set_vel(0, &[3.5]).unwrap();

    let mut t = LangevinThermostat::new(2.0, 0.0, 9, false).unwrap();
    t.apply(&mut obj, 0.1).unwrap();

    assert!((obj.vel_of(0).unwrap()[0] - 3.5).abs() < 1e-12);
}

#[test]
/// Annotation:
/// - Purpose: Executes `langevin_rejects_invalid_dt` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn langevin_rejects_invalid_dt() {
    let mut obj = PhysObj::<1>::empty(1);
    let mut t = LangevinThermostat::new(1.0, 0.5, 1, false).unwrap();

    assert_eq!(
        t.apply(&mut obj, 0.0),
        Err(ThermostatError::InvalidDt { dt: 0.0 })
    );
}

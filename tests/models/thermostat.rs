use physics_in_parallel::models::particles::attrs::ATTR_V;
use physics_in_parallel::models::particles::create_state::create_template;
use physics_in_parallel::models::particles::thermostat::{
    LangevinThermostat, Thermostat, ThermostatError,
};

#[test]
fn langevin_deterministic_for_same_seed_and_state() {
    let mut a = create_template(2, 2).unwrap();
    let mut b = a.clone();

    a.core
        .set_vector_of::<f64>(ATTR_V, 0, &[1.0, -1.0])
        .unwrap();
    a.core.set_vector_of::<f64>(ATTR_V, 1, &[0.5, 2.0]).unwrap();
    b.core
        .set_vector_of::<f64>(ATTR_V, 0, &[1.0, -1.0])
        .unwrap();
    b.core.set_vector_of::<f64>(ATTR_V, 1, &[0.5, 2.0]).unwrap();

    let mut ta = LangevinThermostat::new(1.0, 0.7, 42, false).unwrap();
    let mut tb = LangevinThermostat::new(1.0, 0.7, 42, false).unwrap();

    ta.apply(&mut a, 0.05).unwrap();
    tb.apply(&mut b, 0.05).unwrap();

    assert_eq!(
        a.core.get::<f64>(ATTR_V).unwrap().as_tensor().data,
        b.core.get::<f64>(ATTR_V).unwrap().as_tensor().data
    );
}

#[test]
fn langevin_zero_gamma_keeps_velocity_unchanged() {
    let mut obj = create_template(1, 1).unwrap();
    obj.core.set_vector_of::<f64>(ATTR_V, 0, &[3.5]).unwrap();

    let mut t = LangevinThermostat::new(2.0, 0.0, 9, false).unwrap();
    t.apply(&mut obj, 0.1).unwrap();

    assert!((obj.core.vector_of::<f64>(ATTR_V, 0).unwrap()[0] - 3.5).abs() < 1e-12);
}

#[test]
fn langevin_rejects_invalid_dt() {
    let mut obj = create_template(1, 1).unwrap();
    let mut t = LangevinThermostat::new(1.0, 0.5, 1, false).unwrap();

    assert_eq!(
        t.apply(&mut obj, 0.0),
        Err(ThermostatError::InvalidDt { dt: 0.0 })
    );
}

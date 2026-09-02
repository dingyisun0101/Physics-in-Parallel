use physics_in_parallel::prelude::advanced::PhysObjAdvanced;
use physics_in_parallel::prelude::basic::*;
use physics_in_parallel::prelude::models::*;

#[test]
fn kinetic_energy_uses_inverse_mass_and_alive_mask() {
    let mut obj = create_template(2, 3).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 0, &[3.0, 4.0])
        .unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 1, &[1.0, 0.0])
        .unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 2, &[2.0, 0.0])
        .unwrap();
    obj.set_attribute_vector::<f64>(ATTR_M_INV, 0, &[0.5])
        .unwrap();
    obj.set_attribute_vector::<f64>(ATTR_M_INV, 1, &[1.0])
        .unwrap();
    obj.set_attribute_vector::<f64>(ATTR_M_INV, 2, &[2.0])
        .unwrap();
    set_alive(&mut obj, 1, false).unwrap();

    let ke = KineticEnergyObserver::default().observe(&obj).unwrap();
    assert_eq!(ke, 0.5 * 25.0 / 0.5 + 0.5 * 4.0 / 2.0);

    let ke_all = KineticEnergyObserver::new(ParticleSelection::All)
        .observe(&obj)
        .unwrap();
    assert_eq!(ke_all, ke + 0.5);
}

#[test]
fn temperature_uses_active_particle_degrees_of_freedom() {
    let mut obj = create_template(2, 2).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 0, &[1.0, 0.0])
        .unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 1, &[0.0, 2.0])
        .unwrap();
    set_alive(&mut obj, 1, false).unwrap();

    let temp = TemperatureObserver::default().observe(&obj).unwrap();
    assert_eq!(temp, 2.0 * 0.5 / 2.0);

    let temp_all = TemperatureObserver::new(ParticleSelection::All)
        .observe(&obj)
        .unwrap();
    assert_eq!(temp_all, 2.0 * 2.5 / 4.0);
    assert_eq!(
        TemperatureObserver::new(ParticleSelection::All).selection(),
        ParticleSelection::All
    );
}

#[test]
fn temperature_returns_zero_when_no_particles_are_active() {
    let mut obj = create_template(2, 2).unwrap();
    set_alive(&mut obj, 0, false).unwrap();
    set_alive(&mut obj, 1, false).unwrap();

    let temp = TemperatureObserver::default().observe(&obj).unwrap();
    assert_eq!(temp, 0.0);
}

#[test]
fn observer_reports_invalid_numeric_state() {
    let mut obj = create_template(1, 2).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_M_INV, 0, &[0.0])
        .unwrap();

    assert_eq!(
        KineticEnergyObserver::default().observe(&obj).unwrap_err(),
        ObserveError::InvalidState {
            field: ATTR_M_INV,
            value: 0.0,
        }
    );

    obj.set_attribute_vector::<f64>(ATTR_M_INV, 0, &[1.0])
        .unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 1, &[f64::INFINITY])
        .unwrap();

    assert_eq!(
        KineticEnergyObserver::default().observe(&obj).unwrap_err(),
        ObserveError::InvalidState {
            field: ATTR_V,
            value: f64::INFINITY,
        }
    );
}

#[test]
fn alive_only_ignores_invalid_velocity_on_dead_particle_but_all_reports_it() {
    let mut obj = create_template(1, 2).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 0, &[1.0]).unwrap();
    obj.set_attribute_vector::<f64>(ATTR_V, 1, &[f64::INFINITY])
        .unwrap();
    set_alive(&mut obj, 1, false).unwrap();

    let ke = KineticEnergyObserver::default().observe(&obj).unwrap();
    assert_eq!(ke, 0.5);

    assert_eq!(
        KineticEnergyObserver::new(ParticleSelection::All)
            .observe(&obj)
            .unwrap_err(),
        ObserveError::InvalidState {
            field: ATTR_V,
            value: f64::INFINITY,
        }
    );
}

#[test]
fn observer_reports_shape_errors() {
    let mut obj = create_template(1, 2).unwrap();
    obj.attributes_mut().remove(ATTR_M_INV).unwrap();
    obj.attributes_mut()
        .insert(ATTR_M_INV, VectorList::<f64>::empty(2, 2))
        .unwrap();

    assert_eq!(
        KineticEnergyObserver::default().observe(&obj).unwrap_err(),
        ObserveError::InvalidAttrShape {
            label: ATTR_M_INV,
            expected_dim: 1,
            got_dim: 2,
        }
    );
}

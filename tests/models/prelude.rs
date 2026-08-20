use physics_in_parallel::prelude::models::*;

#[test]
fn models_prelude_exports_common_types() {
    let mut objects = create_template(2, 2).unwrap();
    set_alive(&mut objects, 1, false).unwrap();
    set_rigid(&mut objects, 0, true).unwrap();

    assert!(is_alive(&objects, 0).unwrap());
    assert!(!is_alive(&objects, 1).unwrap());
    assert!(is_rigid(&objects, 0).unwrap());

    let _spring = Spring::new(1.0, 0.5, None).unwrap();
    let _power_law = PowerLawDecay::new(1.0, 2.0, None).unwrap();

    let _ke = KineticEnergyObserver::new(ParticleSelection::AliveOnly)
        .observe(&objects)
        .unwrap();
    let mut integrator = ExplicitEuler;
    integrator.apply(&mut objects, 0.1).unwrap();
}

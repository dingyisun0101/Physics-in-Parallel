use physics_in_parallel::prelude::models::{
    ExplicitEuler, IntegratorError, KineticEnergyObserver, LangevinThermostat, ObserveError,
    Observer, ParticleBoundaryError, ParticleSelection, ParticleStateError, PowerLawDecay,
    PowerLawNetwork, PowerLawNetworkError, Spring, SpringNetwork, SpringNetworkError,
    TemperatureObserver, ThermostatError,
};

fn assert_error<T: std::error::Error + Send + Sync + 'static>() {}
fn assert_send<T: Send>() {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn laws_are_validated_and_read_only() {
    let spring = Spring::new(2.5, 1.25, Some((0.5, 3.0))).unwrap();
    assert_eq!(spring.spring_constant(), 2.5);
    assert_eq!(spring.rest_length(), 1.25);
    assert_eq!(spring.cutoff(), Some((0.5, 3.0)));

    let power = PowerLawDecay::new(-1.0, -2.0, None).unwrap();
    assert_eq!(power.strength(), -1.0);
    assert_eq!(power.exponent(), -2.0);
    assert_eq!(power.range(), None);
}

#[test]
fn network_insertion_is_pair_keyed_and_returns_the_previous_law() {
    let first = Spring::new(1.0, 2.0, None).unwrap();
    let replacement = Spring::new(3.0, 4.0, None).unwrap();
    let mut network = SpringNetwork::new();

    assert_eq!(network.insert((4, 1), first).unwrap(), None);
    assert_eq!(network.minimum_particle_count(), 5);
    assert_eq!(network.insert((1, 4), replacement).unwrap(), Some(first));
    assert_eq!(network.len(), 1);
    assert_eq!(network.get((4, 1)), Some(&replacement));
}

#[test]
fn network_batches_are_transactional() {
    let law = Spring::new(1.0, 1.0, None).unwrap();
    let mut network = SpringNetwork::new();
    network.insert((0, 1), law).unwrap();

    let error = network
        .insert_many([((1, 2), law), ((3, 3), law)])
        .unwrap_err();
    assert!(matches!(
        error,
        SpringNetworkError::SelfPair { particle: 3 }
    ));
    assert_eq!(network.len(), 1);
    assert!(network.get((1, 2)).is_none());
}

#[test]
fn direct_network_serde_rejects_duplicate_pairs() {
    let duplicate = r#"{
        "springs": [
            {"pair": [0, 1], "law": {"k": 1.0, "l_0": 1.0, "cutoff": null}},
            {"pair": [1, 0], "law": {"k": 2.0, "l_0": 1.0, "cutoff": null}}
        ]
    }"#;
    assert!(serde_json::from_str::<SpringNetwork>(duplicate).is_err());
}

#[test]
fn model_traits_and_errors_have_the_approved_concurrency_bounds() {
    assert_send::<ExplicitEuler>();
    assert_send::<LangevinThermostat>();
    assert_send_sync::<KineticEnergyObserver>();
    assert_send_sync::<TemperatureObserver>();

    assert_error::<ParticleStateError>();
    assert_error::<ParticleBoundaryError>();
    assert_error::<IntegratorError>();
    assert_error::<ThermostatError>();
    assert_error::<ObserveError>();
    assert_error::<SpringNetworkError>();
    assert_error::<PowerLawNetworkError>();

    fn accepts_observer<T: Observer>(_observer: T) {}
    accepts_observer(KineticEnergyObserver::new(ParticleSelection::AliveOnly));

    let law = PowerLawDecay::new(1.0, -2.0, None).unwrap();
    let mut network = PowerLawNetwork::new();
    network.insert_all_to_all(3, law).unwrap();
    assert_eq!(network.len(), 3);
}

#[test]
fn integrators_mass_and_thermostat_contracts_cover_both_backends() {
    use physics_in_parallel::prelude::basic::{Backend, ResolvedRng, RngMethod};
    use physics_in_parallel::prelude::models::*;
    for backend in [Backend::Dense, Backend::Sparse] {
        let mut base = create_template(2, 3).unwrap();
        for label in [ATTR_R, ATTR_V, ATTR_A, ATTR_M, ATTR_M_INV] {
            base.attribute_mut::<f64>(label)
                .unwrap()
                .set_backend(backend);
        }
        for i in 0..3 {
            base.set_attribute_vector(ATTR_V, i, &[2.0, 3.0]).unwrap();
            base.set_attribute_vector(ATTR_A, i, &[4.0, -2.0]).unwrap();
        }
        set_alive(&mut base, 1, false).unwrap();
        set_rigid(&mut base, 2, true).unwrap();
        set_mass(&mut base, 0, 2.0).unwrap();
        assert_eq!(
            base.attribute::<f64>(ATTR_M_INV)
                .unwrap()
                .get(0, 0)
                .unwrap(),
            0.5
        );
        assert!(set_mass(&mut base, 0, 0.0).is_err());
        let mut explicit = base.clone();
        ExplicitEuler.apply(&mut explicit, 0.25).unwrap();
        let mut semi = base.clone();
        SemiImplicitEuler.apply(&mut semi, 0.25).unwrap();
        assert_eq!(
            explicit.attribute_vector::<f64>(ATTR_R, 0).unwrap(),
            [0.5, 0.75]
        );
        assert_eq!(
            semi.attribute_vector::<f64>(ATTR_R, 0).unwrap(),
            [0.75, 0.625]
        );
        assert_eq!(
            explicit.attribute_vector::<f64>(ATTR_V, 0).unwrap(),
            [3.0, 2.5]
        );
        for i in [1, 2] {
            assert_eq!(
                explicit.attribute_vector::<f64>(ATTR_R, i).unwrap(),
                base.attribute_vector::<f64>(ATTR_R, i).unwrap()
            );
        }
        let summary = kinetic_summary(&base, ParticleSelection::AliveOnly).unwrap();
        assert_eq!(summary.particle_count, 2);
        assert_eq!(
            summary.energy,
            KineticEnergyObserver::default().observe(&base).unwrap()
        );
        assert_eq!(
            summary.temperature,
            TemperatureObserver::default().observe(&base).unwrap()
        );
        set_alive(&mut base, 1, true).unwrap();
        set_rigid(&mut base, 2, false).unwrap();
        let rng = ResolvedRng::new(9, RngMethod::IndexedSplitMix64);
        for bad in [0, 2] {
            let mut particles = base.clone();
            particles
                .attribute_mut::<f64>(ATTR_M_INV)
                .unwrap()
                .set(bad, 0, -1.0)
                .unwrap();
            let before = particles
                .attribute::<f64>(ATTR_V)
                .unwrap()
                .values()
                .collect::<Vec<_>>();
            let mut thermostat =
                LangevinThermostat::new(1.0, 0.5, rng, ParticleSelection::All).unwrap();
            assert!(thermostat.apply(&mut particles, 0.1).is_err());
            assert_eq!(thermostat.step_counter(), 0);
            assert_eq!(
                particles
                    .attribute::<f64>(ATTR_V)
                    .unwrap()
                    .values()
                    .collect::<Vec<_>>(),
                before
            );
            set_mass(&mut particles, bad, 1.0).unwrap();
            let mut replay = particles.clone();
            let mut restored =
                LangevinThermostat::from_state(1.0, 0.5, rng, 0, ParticleSelection::All).unwrap();
            thermostat.apply(&mut particles, 0.1).unwrap();
            restored.apply(&mut replay, 0.1).unwrap();
            assert_eq!(
                particles
                    .attribute::<f64>(ATTR_V)
                    .unwrap()
                    .values()
                    .collect::<Vec<_>>(),
                replay
                    .attribute::<f64>(ATTR_V)
                    .unwrap()
                    .values()
                    .collect::<Vec<_>>()
            );
        }
    }
}

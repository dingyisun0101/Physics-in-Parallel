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

#[test]
fn neighbor_rebuild_queries_match_brute_force_after_motion_and_death() {
    use physics_in_parallel::prelude::models::*;
    let mut particles = create_template(2, 12).unwrap();
    for i in 0..12 {
        particles
            .set_attribute_vector(ATTR_R, i, &[i as f64 * 0.31, (i % 3) as f64 * 0.2])
            .unwrap();
    }
    let mut neighbors = ParticleNeighborList::from_box(&[1e6, 1e6], 0.75).unwrap();
    assert!(
        neighbors
            .collect_pairs(&particles, ParticleSelection::All)
            .is_err()
    );
    let mut output = Vec::new();
    for step in 0..3 {
        if step > 0 {
            particles
                .set_attribute_vector(ATTR_R, step, &[20.0, 20.0])
                .unwrap();
            set_alive(&mut particles, 7, false).unwrap();
        }
        neighbors
            .rebuild_and_collect_into(&particles, ParticleSelection::AliveOnly, &mut output)
            .unwrap();
        output.sort_unstable();
        let mut expected = Vec::new();
        for i in 0..12 {
            for j in i + 1..12 {
                if !is_alive(&particles, i).unwrap() || !is_alive(&particles, j).unwrap() {
                    continue;
                }
                let a = particles.attribute_vector::<f64>(ATTR_R, i).unwrap();
                let b = particles.attribute_vector::<f64>(ATTR_R, j).unwrap();
                let distance: f64 = a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum();
                if distance > 0.0 && distance < 0.75 * 0.75 {
                    expected.push((i, j));
                }
            }
        }
        assert_eq!(output, expected);
    }
}

#[test]
fn forces_preserve_balance_selection_and_validation_atomicity() {
    use physics_in_parallel::prelude::models::*;
    let mut particles = create_template(1, 2).unwrap();
    particles.set_attribute_vector(ATTR_R, 1, &[2.0]).unwrap();
    set_mass(&mut particles, 0, 2.0).unwrap();
    let mut springs = SpringNetwork::new();
    springs
        .insert((0, 1), Spring::new(2.0, 1.0, None).unwrap())
        .unwrap();
    springs
        .apply(&mut particles, ParticleSelection::AliveOnly)
        .unwrap();
    let a = particles.attribute::<f64>(ATTR_A).unwrap();
    assert_eq!(a.get(0, 0).unwrap(), 1.0);
    assert_eq!(a.get(1, 0).unwrap(), -2.0);
    set_rigid(&mut particles, 0, true).unwrap();
    springs
        .apply(&mut particles, ParticleSelection::AliveOnly)
        .unwrap();
    assert_eq!(
        particles
            .attribute::<f64>(ATTR_A)
            .unwrap()
            .get(0, 0)
            .unwrap(),
        1.0
    );
    let before = particles
        .attribute::<f64>(ATTR_A)
        .unwrap()
        .values()
        .collect::<Vec<_>>();
    springs
        .insert((1, 2), Spring::new(1.0, 1.0, None).unwrap())
        .unwrap();
    assert!(
        springs
            .apply(&mut particles, ParticleSelection::AliveOnly)
            .is_err()
    );
    assert_eq!(
        particles
            .attribute::<f64>(ATTR_A)
            .unwrap()
            .values()
            .collect::<Vec<_>>(),
        before
    );
}

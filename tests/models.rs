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

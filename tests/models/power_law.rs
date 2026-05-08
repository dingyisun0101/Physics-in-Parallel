use physics_in_parallel::models::laws::{PowerLawDecay, PowerLawError};
use physics_in_parallel::models::particles::interactions::power_law::{
    PowerLawNetwork, PowerLawNetworkError,
};

#[test]
fn empty_add_get_remove_roundtrip() {
    let mut network = PowerLawNetwork::empty();
    assert!(network.is_empty());

    let id = network
        .add_power_law((7, 2), 6.67, -2.0, Some((0.1, 10.0)))
        .unwrap();
    assert_eq!(id, 0);
    assert_eq!(network.len(), 1);

    let law = network.get_power_law((2, 7)).unwrap().unwrap();
    assert_eq!(law.k, 6.67);
    assert_eq!(law.alpha, -2.0);
    assert_eq!(law.range, Some((0.1, 10.0)));

    let removed = network.remove_power_law((2, 7)).unwrap().unwrap();
    assert_eq!(removed.k, 6.67);
    assert_eq!(removed.alpha, -2.0);
    assert_eq!(removed.range, Some((0.1, 10.0)));
    assert!(network.is_empty());
}

#[test]
fn add_twice_same_pair_overwrites_payload() {
    let mut network = PowerLawNetwork::empty();
    let id0 = network.add_power_law((0, 3), 1.0, -1.0, None).unwrap();
    let id1 = network
        .add_power_law((3, 0), 3.0, 2.0, Some((0.5, 8.0)))
        .unwrap();

    assert_eq!(id0, id1);
    assert_eq!(network.len(), 1);

    let law = network.get_power_law((0, 3)).unwrap().unwrap();
    assert_eq!(law.k, 3.0);
    assert_eq!(law.alpha, 2.0);
    assert_eq!(law.range, Some((0.5, 8.0)));
}

#[test]
fn remove_nonexistent_pair_returns_none() {
    let mut network = PowerLawNetwork::empty();
    assert!(network.remove_power_law((10, 11)).unwrap().is_none());

    network.add_power_law((0, 1), 2.0, -2.0, None).unwrap();
    assert!(network.remove_power_law((0, 2)).unwrap().is_none());
}

#[test]
fn invalid_power_law_parameters_are_rejected() {
    let mut network = PowerLawNetwork::empty();

    assert!(matches!(
        network
            .add_power_law((0, 1), f64::NAN, 1.0, None)
            .unwrap_err(),
        PowerLawNetworkError::Law(PowerLawError::InvalidStrength { k }) if k.is_nan()
    ));
    assert_eq!(
        network
            .add_power_law((0, 1), 1.0, f64::INFINITY, None)
            .unwrap_err(),
        PowerLawNetworkError::Law(PowerLawError::InvalidExponent {
            alpha: f64::INFINITY
        })
    );
    assert_eq!(
        network
            .add_power_law((0, 1), 1.0, -2.0, Some((5.0, 1.0)))
            .unwrap_err(),
        PowerLawNetworkError::Law(PowerLawError::InvalidRange { min: 5.0, max: 1.0 })
    );
}

#[test]
fn add_payload_validates_prebuilt_payload_and_mutation_path() {
    let mut network = PowerLawNetwork::empty();
    let payload = PowerLawDecay::new(2.0, -3.0, Some((0.2, 4.0))).unwrap();

    let id = network.add_payload((2, 4), payload).unwrap();
    assert_eq!(id, 0);

    let law = network.get_power_law_mut((4, 2)).unwrap().unwrap();
    law.k = 5.0;

    assert_eq!(network.get_power_law((2, 4)).unwrap().unwrap().k, 5.0);
    assert_eq!(network.interaction().len(), 1);

    let invalid = PowerLawDecay {
        k: 1.0,
        alpha: f64::NAN,
        range: None,
    };
    assert!(matches!(
        network.add_payload((0, 1), invalid).unwrap_err(),
        PowerLawNetworkError::Law(PowerLawError::InvalidExponent { .. })
    ));
}

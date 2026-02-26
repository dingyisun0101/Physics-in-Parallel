use physics_in_parallel::models::particles::interactions::power_law::PowerLawNetwork;

#[test]
fn empty_add_get_delete_roundtrip() {
    let mut network = PowerLawNetwork::empty();
    assert!(network.is_empty());

    let edge = network.add((7, 2), 6.67, -2.0, Some((10.0, 0.1))).unwrap();
    assert_eq!(edge, 0);
    assert_eq!(network.len(), 1);

    let law = network.get_power_law((2, 7)).unwrap().unwrap();
    assert_eq!(law.k, 6.67);
    assert_eq!(law.alpha, -2.0);
    assert_eq!(law.range, Some((10.0, 0.1)));

    let deleted = network.delete((2, 7)).unwrap().unwrap();
    assert_eq!(deleted.k, 6.67);
    assert_eq!(deleted.alpha, -2.0);
    assert_eq!(deleted.range, Some((10.0, 0.1)));
    assert!(network.is_empty());
}

#[test]
fn add_twice_same_pair_overwrites_payload() {
    let mut network = PowerLawNetwork::empty();
    let e0 = network.add((0, 3), 1.0, -1.0, None).unwrap();
    let e1 = network.add((3, 0), 3.0, 2.0, Some((8.0, 0.5))).unwrap();

    assert_eq!(e0, e1);
    assert_eq!(network.len(), 1);

    let law = network.get_power_law((0, 3)).unwrap().unwrap();
    assert_eq!(law.k, 3.0);
    assert_eq!(law.alpha, 2.0);
    assert_eq!(law.range, Some((8.0, 0.5)));
}

#[test]
fn delete_nonexistent_pair_returns_none() {
    let mut network = PowerLawNetwork::empty();
    assert!(network.delete((10, 11)).unwrap().is_none());

    network.add((0, 1), 2.0, -2.0, None).unwrap();
    assert!(network.delete((0, 2)).unwrap().is_none());
}

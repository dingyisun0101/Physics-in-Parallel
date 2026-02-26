use physics_in_parallel::models::particles::interactions::spring_network::SpringNetwork;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn empty_add_get_remove_roundtrip() {
    let mut network = SpringNetwork::empty();
    assert!(network.is_empty());

    let edge = network
        .add_spring((3, 1), 12.0, 0.7, Some((0.2, 2.5)))
        .unwrap();
    assert_eq!(edge, 0);
    assert_eq!(network.len(), 1);

    let spring = network.get_spring((1, 3)).unwrap().unwrap();
    assert_eq!(spring.k, 12.0);
    assert_eq!(spring.l_0, 0.7);
    assert_eq!(spring.cutoff, Some((0.2, 2.5)));

    let removed = network.remove_spring((1, 3)).unwrap().unwrap();
    assert_eq!(removed.k, 12.0);
    assert_eq!(removed.l_0, 0.7);
    assert_eq!(removed.cutoff, Some((0.2, 2.5)));
    assert!(network.is_empty());
}

#[test]
fn add_twice_same_pair_overwrites_payload() {
    let mut network = SpringNetwork::empty();
    let e0 = network.add_spring((0, 2), 10.0, 1.0, None).unwrap();
    let e1 = network
        .add_spring((2, 0), 20.0, 1.5, Some((0.1, 3.0)))
        .unwrap();

    assert_eq!(e0, e1);
    assert_eq!(network.len(), 1);

    let spring = network.get_spring((0, 2)).unwrap().unwrap();
    assert_eq!(spring.k, 20.0);
    assert_eq!(spring.l_0, 1.5);
    assert_eq!(spring.cutoff, Some((0.1, 3.0)));
}

#[test]
fn remove_nonexistent_pair_returns_none() {
    let mut network = SpringNetwork::empty();
    assert!(network.remove_spring((10, 11)).unwrap().is_none());

    network.add_spring((0, 1), 4.0, 1.0, None).unwrap();
    assert!(network.remove_spring((0, 2)).unwrap().is_none());
}

#[test]
fn par_iter_springs_visits_all_active_springs() {
    let mut network = SpringNetwork::empty();
    network.add_spring((0, 1), 2.0, 1.0, None).unwrap();
    network.add_spring((1, 2), 3.0, 1.5, None).unwrap();
    network.add_spring((0, 2), 4.0, 2.0, None).unwrap();

    let count = AtomicUsize::new(0);
    let i_sum = AtomicUsize::new(0);
    let j_sum = AtomicUsize::new(0);

    network.par_iter_springs(|i, j, spring| {
        assert!(i < j);
        assert!(spring.k > 0.0);
        count.fetch_add(1, Ordering::Relaxed);
        i_sum.fetch_add(i, Ordering::Relaxed);
        j_sum.fetch_add(j, Ordering::Relaxed);
    });

    assert_eq!(count.load(Ordering::Relaxed), 3);
    assert_eq!(i_sum.load(Ordering::Relaxed), 1);
    assert_eq!(j_sum.load(Ordering::Relaxed), 5);
}

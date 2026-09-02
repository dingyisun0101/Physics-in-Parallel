use std::sync::atomic::{AtomicUsize, Ordering};

use physics_in_parallel::prelude::advanced::PhysObjAdvanced;
use physics_in_parallel::prelude::basic::VectorList;
use physics_in_parallel::prelude::models::*;

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
fn capacity_constructor_and_batch_payload_insert_work() {
    let mut network = SpringNetwork::with_capacity(4, 3);
    let spring = Spring::new(5.0, 1.2, None).unwrap();

    network
        .add_springs_payload(&[(0, 1), (2, 3), (1, 3)], spring)
        .unwrap();

    assert_eq!(network.len(), 3);
    assert_eq!(network.get_spring((1, 0)).unwrap().unwrap().k, 5.0);
    assert_eq!(network.get_spring((3, 2)).unwrap().unwrap().l_0, 1.2);
    assert!(network.get_spring((0, 2)).unwrap().is_none());
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

#[test]
fn records_and_serde_preserve_bound_and_stable_active_order() {
    let mut network = SpringNetwork::with_capacity(8, 3);
    network.add_spring((4, 1), 2.0, 1.0, None).unwrap();
    network
        .add_spring((7, 2), 3.0, 1.5, Some((0.1, 4.0)))
        .unwrap();
    network.remove_spring((1, 4)).unwrap();
    network.add_spring((6, 0), 5.0, 2.0, None).unwrap();

    let records = network.records().unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!((records[0].i, records[0].j), (0, 6));
    assert_eq!((records[1].i, records[1].j), (2, 7));

    let json = serde_json::to_string(&network).unwrap();
    let restored: SpringNetwork = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.num_particles(), 8);
    assert_eq!(restored.records().unwrap(), records);

    let empty = SpringNetwork::with_capacity(12, 0);
    let restored_empty: SpringNetwork =
        serde_json::from_str(&serde_json::to_string(&empty).unwrap()).unwrap();
    assert_eq!(restored_empty.num_particles(), 12);
    assert!(restored_empty.is_empty());
}

#[test]
fn from_records_rejects_invalid_network_state() {
    let spring = Spring::new(1.0, 1.0, None).unwrap();

    assert_eq!(
        SpringNetwork::from_records(3, [SpringRecord { i: 1, j: 1, spring }],).unwrap_err(),
        SpringNetworkError::SelfPair { particle: 1 }
    );
    assert_eq!(
        SpringNetwork::from_records(3, [SpringRecord { i: 0, j: 3, spring }],).unwrap_err(),
        SpringNetworkError::ParticleOutOfBounds {
            particle: 3,
            num_particles: 3,
        }
    );
    assert_eq!(
        SpringNetwork::from_records(
            3,
            [
                SpringRecord { i: 0, j: 2, spring },
                SpringRecord { i: 2, j: 0, spring },
            ],
        )
        .unwrap_err(),
        SpringNetworkError::DuplicatePair { i: 0, j: 2 }
    );

    let invalid = SpringRecord {
        i: 0,
        j: 1,
        spring: Spring {
            k: f64::NAN,
            l_0: 1.0,
            cutoff: None,
        },
    };
    assert!(matches!(
        SpringNetwork::from_records(2, [invalid]).unwrap_err(),
        SpringNetworkError::Law(SpringLawError::InvalidSpringConstant { .. })
    ));
}

#[test]
fn spring_network_serde_rejects_malformed_records() {
    let duplicate = r#"{
        "num_particles": 3,
        "springs": [
            {"i": 0, "j": 2, "spring": {"k": 1.0, "l_0": 1.0, "cutoff": null}},
            {"i": 2, "j": 0, "spring": {"k": 2.0, "l_0": 1.0, "cutoff": null}}
        ]
    }"#;
    assert!(
        serde_json::from_str::<SpringNetwork>(duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate spring pair")
    );

    let unknown = r#"{"num_particles":2,"springs":[],"implementation_detail":"forbidden"}"#;
    assert!(serde_json::from_str::<SpringNetwork>(unknown).is_err());
}

#[test]
fn invalid_spring_parameters_are_rejected() {
    let mut network = SpringNetwork::empty();

    assert!(matches!(
        network.add_spring((0, 1), f64::NAN, 1.0, None).unwrap_err(),
        SpringNetworkError::Law(SpringLawError::InvalidSpringConstant { .. })
    ));
    assert!(matches!(
        network.add_spring((0, 1), 1.0, -1.0, None).unwrap_err(),
        SpringNetworkError::Law(SpringLawError::InvalidRestLength { l_0: -1.0 })
    ));
    assert_eq!(
        network
            .add_spring((0, 1), 1.0, 1.0, Some((2.0, 1.0)))
            .unwrap_err(),
        SpringNetworkError::Law(SpringLawError::InvalidCutoff { min: 2.0, max: 1.0 })
    );
}

#[test]
fn hooke_acceleration_two_particle_sign_and_additive_semantics() {
    let mut objects = create_template(1, 2).unwrap();
    objects
        .set_attribute_vector::<f64>(ATTR_R, 0, &[0.0])
        .unwrap();
    objects
        .set_attribute_vector::<f64>(ATTR_R, 1, &[2.0])
        .unwrap();
    objects
        .set_attribute_vector::<f64>(ATTR_A, 0, &[0.5])
        .unwrap();
    objects
        .set_attribute_vector::<f64>(ATTR_A, 1, &[1.0])
        .unwrap();

    let mut network = SpringNetwork::empty();
    network.add_spring((0, 1), 4.0, 1.0, None).unwrap();
    network
        .apply_hooke_acceleration(&mut objects, ParticleSelection::All)
        .unwrap();

    let a = objects.attribute::<f64>(ATTR_A).unwrap();
    assert_eq!(a.get(0, 0), 4.5);
    assert_eq!(a.get(1, 0), -3.0);
}

#[test]
fn hooke_acceleration_respects_rigid_dead_cutoff_and_equal_position_cases() {
    let mut objects = create_template(1, 4).unwrap();

    objects
        .set_attribute_vector::<f64>(ATTR_R, 0, &[0.0])
        .unwrap();
    objects
        .set_attribute_vector::<f64>(ATTR_R, 1, &[2.0])
        .unwrap();
    objects
        .set_attribute_vector::<f64>(ATTR_R, 2, &[4.0])
        .unwrap();
    objects
        .set_attribute_vector::<f64>(ATTR_R, 3, &[4.0])
        .unwrap();

    set_rigid(&mut objects, 0, true).unwrap();
    set_rigid(&mut objects, 1, false).unwrap();
    set_rigid(&mut objects, 2, false).unwrap();
    set_rigid(&mut objects, 3, false).unwrap();

    set_alive(&mut objects, 0, true).unwrap();
    set_alive(&mut objects, 1, true).unwrap();
    set_alive(&mut objects, 2, false).unwrap();
    set_alive(&mut objects, 3, true).unwrap();

    let mut network = SpringNetwork::empty();
    network.add_spring((0, 1), 4.0, 1.0, None).unwrap();
    network
        .add_spring((1, 2), 10.0, 1.0, Some((0.1, 0.5)))
        .unwrap();
    network.add_spring((2, 3), 10.0, 1.0, None).unwrap();

    network
        .apply_hooke_acceleration(&mut objects, ParticleSelection::AliveOnly)
        .unwrap();

    let a = objects.attribute::<f64>(ATTR_A).unwrap();
    assert_eq!(a.get(0, 0), 0.0, "rigid endpoint should not be updated");
    assert_eq!(a.get(1, 0), -4.0);
    assert_eq!(a.get(2, 0), 0.0, "dead/cutoff/equal-position cases skipped");
    assert_eq!(a.get(3, 0), 0.0);
}

#[test]
fn hooke_acceleration_reports_invalid_shapes_and_inverse_mass() {
    let mut bad_mass = create_template(1, 2).unwrap();
    bad_mass
        .set_attribute_vector::<f64>(ATTR_M_INV, 1, &[-1.0])
        .unwrap();
    let mut network = SpringNetwork::empty();
    network.add_spring((0, 1), 1.0, 1.0, None).unwrap();
    assert_eq!(
        network
            .apply_hooke_acceleration(&mut bad_mass, ParticleSelection::All)
            .unwrap_err(),
        SpringNetworkError::InvalidInverseMass {
            index: 1,
            value: -1.0,
        }
    );

    let mut bad_accel = create_template(1, 2).unwrap();
    bad_accel.attributes_mut().remove(ATTR_A).unwrap();
    bad_accel
        .attributes_mut()
        .insert(ATTR_A, VectorList::<f64>::empty(2, 2))
        .unwrap();
    assert_eq!(
        network
            .apply_hooke_acceleration(&mut bad_accel, ParticleSelection::All)
            .unwrap_err(),
        SpringNetworkError::InvalidAttrShape {
            label: ATTR_A,
            expected_dim: 1,
            got_dim: 2,
        }
    );
}

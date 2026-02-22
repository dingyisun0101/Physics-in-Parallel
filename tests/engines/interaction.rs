use physics_in_parallel::engines::soa::interaction::{DirectionMode, InteractionError};
use physics_in_parallel::engines::soa::{Interaction, Topology};

#[test]
fn topology_undirected_rejects_noncanonical_order() {
    let mut topo = Topology::new(4);

    let err = topo.insert(&[2, 1]).unwrap_err();
    assert!(matches!(
        err,
        InteractionError::InvalidUndirectedOrder { .. }
    ));

    let edge = topo.insert(&[1, 2]).unwrap();
    assert_eq!(topo.active_count(), 1);
    assert_eq!(
        topo.edge_key(edge).unwrap().nodes.as_ref(),
        [1, 2].as_slice()
    );
}

#[test]
fn topology_reuses_freed_slot() {
    let mut topo = Topology::new(4);
    let e0 = topo.insert(&[0, 1]).unwrap();
    assert_eq!(e0, 0);

    let removed = topo.remove(&[0, 1]).unwrap().unwrap();
    assert_eq!(removed, e0);

    let e1 = topo.insert(&[1, 3]).unwrap();
    assert_eq!(e1, e0);
}

#[test]
fn interaction_payload_roundtrip_and_parallel_mutation() {
    let mut table = Interaction::<i64>::new(5, DirectionMode::Directed);

    let a = table.insert(&[2, 1], 10).unwrap();
    let b = table.insert(&[4, 0, 3], 20).unwrap();
    assert_ne!(a, b);

    assert_eq!(*table.get(&[2, 1]).unwrap().unwrap(), 10);
    *table.get_mut(&[2, 1]).unwrap().unwrap() += 5;
    assert_eq!(*table.get(&[2, 1]).unwrap().unwrap(), 15);

    table.par_for_each_active_payload_mut(|_, payload| {
        *payload += 1;
    });

    assert_eq!(*table.get(&[2, 1]).unwrap().unwrap(), 16);
    assert_eq!(*table.get(&[4, 0, 3]).unwrap().unwrap(), 21);

    let removed = table.remove(&[2, 1]).unwrap().unwrap();
    assert_eq!(removed.1, 16);
    assert!(table.get(&[2, 1]).unwrap().is_none());
}

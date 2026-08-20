use physics_in_parallel::prelude::advanced::{
    Interaction, InteractionError, InteractionOrder, InteractionTopology,
};

#[test]
fn topology_unordered_canonicalizes_node_order() {
    let mut topo = InteractionTopology::new(4);

    let id = topo.add(&[2, 1]).unwrap();
    let id_2 = topo.add(&[1, 2]).unwrap();
    assert_eq!(id, id_2);
    assert_eq!(topo.len(), 1);
    assert_eq!(topo.nodes_of(id).unwrap().nodes.as_ref(), [1, 2].as_slice());
}

#[test]
fn topology_reuses_freed_id() {
    let mut topo = InteractionTopology::new(4);
    let id0 = topo.add(&[0, 1]).unwrap();
    assert_eq!(id0, 0);

    let removed = topo.remove(&[0, 1]).unwrap().unwrap();
    assert_eq!(removed, id0);

    let id1 = topo.add(&[1, 3]).unwrap();
    assert_eq!(id1, id0);
}

#[test]
fn topology_ordered_keeps_node_order_distinct() {
    let mut topo = InteractionTopology::with_order(4, InteractionOrder::Ordered);

    let forward = topo.add(&[1, 2]).unwrap();
    let reverse = topo.add(&[2, 1]).unwrap();

    assert_ne!(forward, reverse);
    assert_eq!(topo.len(), 2);
    assert_eq!(
        topo.nodes_of(forward).unwrap().nodes.as_ref(),
        [1, 2].as_slice()
    );
    assert_eq!(
        topo.nodes_of(reverse).unwrap().nodes.as_ref(),
        [2, 1].as_slice()
    );
}

#[test]
fn topology_rejects_empty_nodes() {
    let mut topo = InteractionTopology::new(4);

    assert_eq!(topo.add(&[]).unwrap_err(), InteractionError::EmptyNodes);
}

#[test]
fn topology_set_order_rebuilds_or_reports_collision() {
    let mut safe = InteractionTopology::with_order(4, InteractionOrder::Ordered);
    let id = safe.add(&[2, 1]).unwrap();
    safe.set_order(InteractionOrder::Unordered).unwrap();
    assert_eq!(safe.order(), InteractionOrder::Unordered);
    assert_eq!(safe.id_of(&[1, 2]).unwrap(), Some(id));
    assert_eq!(safe.nodes_of(id).unwrap().nodes.as_ref(), [1, 2].as_slice());

    let mut colliding = InteractionTopology::with_order(4, InteractionOrder::Ordered);
    let _ = colliding.add(&[1, 2]).unwrap();
    let _ = colliding.add(&[2, 1]).unwrap();
    let err = colliding
        .set_order(InteractionOrder::Unordered)
        .unwrap_err();
    assert!(matches!(err, InteractionError::OrderChangeCollision { .. }));
    assert_eq!(colliding.order(), InteractionOrder::Ordered);
}

#[test]
fn interaction_payload_roundtrip_and_parallel_mutation() {
    let mut table = Interaction::<i64>::new(5, InteractionOrder::Ordered);

    let a = table.set(&[2, 1], 10).unwrap();
    let b = table.set(&[4, 0, 3], 20).unwrap();
    assert_ne!(a, b);

    assert_eq!(*table.get(&[2, 1]).unwrap().unwrap(), 10);
    *table.get_mut(&[2, 1]).unwrap().unwrap() += 5;
    assert_eq!(*table.get(&[2, 1]).unwrap().unwrap(), 15);

    table.par_for_each_payload_mut(|_, payload| {
        *payload += 1;
    });

    assert_eq!(*table.get(&[2, 1]).unwrap().unwrap(), 16);
    assert_eq!(*table.get(&[4, 0, 3]).unwrap().unwrap(), 21);

    let removed = table.remove(&[2, 1]).unwrap().unwrap();
    assert_eq!(removed.1, 16);
    assert!(table.get(&[2, 1]).unwrap().is_none());
}

#[test]
fn interaction_pair_helpers_use_order_rules() {
    let mut table = Interaction::<i64>::new(4, InteractionOrder::Unordered);

    let id = table.set_pair(3, 1, 8).unwrap();
    assert_eq!(table.topology().id_of_pair(1, 3).unwrap(), Some(id));
    assert_eq!(*table.get_pair(1, 3).unwrap().unwrap(), 8);

    *table.get_pair_mut(3, 1).unwrap().unwrap() = 11;
    assert_eq!(*table.get_pair(1, 3).unwrap().unwrap(), 11);

    let removed = table.remove_pair(1, 3).unwrap().unwrap();
    assert_eq!(removed, (id, 11));
    assert!(table.is_empty());
}

#[test]
fn interaction_strict_resize_and_prune_have_separate_meanings() {
    let mut table = Interaction::<i64>::new(5, InteractionOrder::Unordered);
    let valid = table.set_pair(0, 1, 10).unwrap();
    let invalid_after_shrink = table.set_pair(3, 4, 20).unwrap();

    let err = table.set_n_objects(4).unwrap_err();
    assert!(matches!(
        err,
        InteractionError::ObjectCountWouldInvalidate {
            n_objects: 4,
            id,
            obj: 4,
        } if id == invalid_after_shrink
    ));

    let removed = table.prune_n_objects(4);
    assert_eq!(removed, vec![(invalid_after_shrink, 20)]);
    assert_eq!(table.n_objects(), 4);
    assert_eq!(table.len(), 1);
    assert_eq!(*table.get_pair(0, 1).unwrap().unwrap(), 10);
    assert_eq!(
        table.topology().nodes_of(valid).unwrap().nodes.as_ref(),
        [0, 1]
    );
}

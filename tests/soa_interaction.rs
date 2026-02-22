use physics_in_parallel::engines::soa_engine::interaction::{
    Interaction, InteractionError, PairTopology, SpringInteraction,
};
use physics_in_parallel::engines::soa_engine::phys_obj::PhysObj;

#[test]
fn pair_topology_insert_delete_and_adj_matrix() {
    let mut topo = PairTopology::new(4);

    let e01 = topo.insert(0, 1).unwrap();
    let e23 = topo.insert(2, 3).unwrap();
    let e12 = topo.insert(1, 2).unwrap();

    assert_eq!(topo.active_count(), 3);
    assert!(topo.contains_pair(1, 0));
    assert_eq!(topo.index_of(0, 1), Some(e01));
    assert_eq!(topo.index_of(3, 2), Some(e23));
    assert_eq!(topo.index_of(2, 1), Some(e12));

    let a = topo.to_adj_matrix();
    assert_eq!(a.get(0, 1), 1);
    assert_eq!(a.get(1, 0), 1);
    assert_eq!(a.get(2, 3), 1);
    assert_eq!(a.get(3, 2), 1);
    assert_eq!(a.get(1, 2), 1);
    assert_eq!(a.get(2, 1), 1);
    assert_eq!(a.get(0, 0), 0);
    assert_eq!(a.get(3, 3), 0);
    assert_eq!(a.get(0, 3), 0);

    topo.delete(2, 3).unwrap();
    assert_eq!(topo.active_count(), 2);
    assert!(!topo.contains_pair(2, 3));
    assert!(!topo.is_active_index(e23));
}

#[test]
fn pair_topology_rejects_invalid_edges() {
    let mut topo = PairTopology::new(3);

    assert_eq!(
        topo.insert(1, 1),
        Err(InteractionError::SelfEdge { obj: 1 })
    );

    assert_eq!(
        topo.insert(0, 9),
        Err(InteractionError::InvalidObjId {
            obj: 9,
            n_objects: 3
        })
    );
}

#[test]
fn spring_interaction_accumulate_acc_1d_two_body() {
    let mut obj = PhysObj::<1>::empty(2);
    obj.set_pos(0, &[0.0]).unwrap();
    obj.set_pos(1, &[2.0]).unwrap();
    obj.set_inv_mass(0, 1.0).unwrap();
    obj.set_inv_mass(1, 1.0).unwrap();
    obj.clear_acc();

    let mut spring = SpringInteraction::new(2);
    spring.add_edge(0, 1, 2.0, 1.0).unwrap();

    spring.accumulate_acc(&mut obj).unwrap();

    // norm = 2, l0 = 1, k = 2 -> f_mag = -2
    // u(0->1) = +1 => a0 += -2, a1 -= -2 = +2
    assert_eq!(obj.acc_of(0).unwrap(), [-2.0].as_slice());
    assert_eq!(obj.acc_of(1).unwrap(), [2.0].as_slice());
}

#[test]
fn spring_interaction_add_edge_updates_existing_params() {
    let mut spring = SpringInteraction::new(3);
    let e0 = spring.add_edge(0, 2, 3.0, 1.2).unwrap();
    let e1 = spring.add_edge(2, 0, 4.0, 2.2).unwrap();

    assert_eq!(e0, e1);
    assert_eq!(spring.active_count(), 1);

    let (k, l0) = spring.edge_params(e0).unwrap();
    assert_eq!(k, 4.0);
    assert_eq!(l0, 2.2);
}

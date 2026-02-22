use physics_in_parallel::engines::prelude::*;
use physics_in_parallel::engines::soa::interaction::DirectionMode;

#[test]
fn engines_prelude_exports_core_soa_types() {
    let mut topo = Topology::new(3);
    let e = topo.insert(&[0, 2]).unwrap();
    assert_eq!(topo.edge_key(e).unwrap().nodes.as_ref(), [0, 2].as_slice());

    let mut interactions = Interaction::<i32>::new(3, DirectionMode::Undirected);
    interactions.insert(&[0, 1], 7).unwrap();
    assert_eq!(*interactions.get(&[0, 1]).unwrap().unwrap(), 7);

    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 2, 1).unwrap();
    let obj = PhysObj {
        meta: AttrsMeta::empty(),
        core,
    };
    assert_eq!(obj.core.dim_of("r").unwrap(), 2);
}

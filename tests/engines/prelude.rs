use physics_in_parallel::prelude::advanced::*;
use physics_in_parallel::prelude::models::PhysObj;

#[test]
fn advanced_prelude_exports_core_soa_types() {
    let mut topo = InteractionTopology::new(3);
    let id = topo.add(&[0, 2]).unwrap();
    assert_eq!(topo.nodes_of(id).unwrap().nodes.as_ref(), [0, 2].as_slice());

    let mut interactions = Interaction::<i32>::new(3, InteractionOrder::Unordered);
    interactions.set(&[0, 1], 7).unwrap();
    assert_eq!(*interactions.get(&[0, 1]).unwrap().unwrap(), 7);

    let mut core = AttrsCore::empty();
    core.allocate::<f64>("r", 2, 1).unwrap();
    let obj = PhysObj::from_raw_parts(AttrsMeta::empty(), core);
    assert_eq!(obj.attributes().dim_of("r").unwrap(), 2);
}

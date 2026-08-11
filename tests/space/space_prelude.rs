use physics_in_parallel::space::prelude::*;

#[test]
fn space_prelude_compiles_for_common_types() {
    let cfg = SquareLatticeConfig::new(&vec![4; 1], BoundaryCondition::Periodic, None);
    let mut g =
        SquareLattice::<usize>::new(cfg, SquareLatticeInitMethod::Uniform { val: 1 }).unwrap();
    g.set(&[1], 0);
    assert_eq!(*g.get(&[1]), 0);

    let mut rpg = RandPairGenerator::new(
        &[4],
        KernelType::NearestNeighbor { d: 1 },
        8,
        SourceMode::Origin,
        RandomKey::new(Some(7)),
    )
    .expect("valid pair generator");
    rpg.refresh_at(0);
    assert_eq!(rpg.sources().shape(), [8, 1]);
}

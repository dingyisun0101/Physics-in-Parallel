use physics_in_parallel::space::prelude::*;

#[test]
fn space_prelude_compiles_for_common_types() {
    let cfg = SquareLatticeConfig::new(&vec![4; 1], BoundaryCondition::Periodic);
    let mut g = SquareLattice::<usize>::new(cfg, SquareLatticeInitMethod::Uniform { val: 1 });
    g.set_vacant(&[1]);
    assert!(g.is_vacant(&[1]));

    let mut rpg = RandPairGenerator::new(
        &[4],
        KernelType::NearestNeighbor { d: 1 },
        8,
        SourceMode::Origin,
        PairRandomKey::new(7),
    )
    .expect("valid pair generator");
    rpg.refresh_at(0);
    assert_eq!(rpg.sources().shape(), [8, 1]);
}

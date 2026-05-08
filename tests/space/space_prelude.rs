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
        Some(1),
    );
    rpg.refresh();
    assert_eq!(rpg.sources().shape(), [8, 1]);
}

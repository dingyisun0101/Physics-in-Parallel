use physics_in_parallel::prelude::basic::*;

#[test]
fn space_prelude_compiles_for_common_types() {
    let cfg = SquareLatticeConfig::try_new(&[4; 1], BoundaryCondition::Periodic, None).unwrap();
    let mut g =
        SquareLattice::<usize>::new(cfg, SquareLatticeInitMethod::Uniform { val: 1 }).unwrap();
    g.set(&[1], 0);
    assert_eq!(*g.get(&[1]), 0);

    let mut rpg = PairGenerator::new(
        &[4],
        PairGeneratorConfig::kernel(
            KernelType::NearestNeighbor { d: 1 },
            8,
            SourceMode::Origin,
            RngConfig::new(Some(7), None),
        ),
    )
    .expect("valid pair generator");
    rpg.refresh_at(0);
    assert_eq!(rpg.sources().shape(), [8, 1]);
}

use physics_in_parallel::space::prelude::*;

#[test]
/// Annotation:
/// - Purpose: Executes `space_prelude_compiles_for_common_types` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn space_prelude_compiles_for_common_types() {
    let cfg = GridConfig::new(1, 4, true);
    let mut g = Grid::<usize>::new(cfg, GridInitMethod::Uniform { val: 1 });
    g.set_vacant(&[1]);
    assert!(g.is_vacant(&[1]));

    let mut rpg = RandPairGenerator::new(
        KernelType::NearestNeighbor { d: 1 },
        1,
        8,
        None,
        Some(1),
    );
    rpg.refresh();
    assert_eq!(rpg.sources().shape(), [1, 8]);
}

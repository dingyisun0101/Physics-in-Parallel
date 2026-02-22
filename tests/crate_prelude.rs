use physics_in_parallel::prelude::*;

#[test]
/// Annotation:
/// - Purpose: Executes `crate_prelude_compiles_for_cross_module_basics` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn crate_prelude_compiles_for_cross_module_basics() {
    let mut t = Tensor::<f64, DenseBackend>::empty(&[2]);
    t.set(&[0], 1.0);

    let mut obj = PhysObj::<1>::empty(1);
    obj.set_vel(0, &[2.0]).unwrap();

    let cfg = GridConfig::new(1, 2, true);
    let grid = Grid::<usize>::new(cfg, GridInitMethod::Uniform { val: 1 });

    assert_eq!(t.get(&[0]), 1.0);
    assert_eq!(obj.vel_of(0).unwrap(), [2.0].as_slice());
    assert_eq!(grid.data().len(), 2);
}

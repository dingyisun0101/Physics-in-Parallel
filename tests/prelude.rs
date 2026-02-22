use physics_in_parallel::math::prelude::*;

#[test]
/// Annotation:
/// - Purpose: Executes `prelude_compiles_for_common_types` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn prelude_compiles_for_common_types() {
    let mut t = Tensor::<f64, DenseBackend>::empty(&[2, 2]);
    t.set(&[0, 1], 3.0);
    assert_eq!(t.get(&[0, 1]), 3.0);

    let mut m = Matrix::<f64>::empty(2, 2);
    m.set(1, 1, 4.0);
    assert_eq!(m.get(1, 1), 4.0);

    let mut vl = VectorList::<f64>::empty(3, 2);
    vl.set(1, 2, 5.0);
    assert_eq!(vl.get(1, 2), 5.0);
}


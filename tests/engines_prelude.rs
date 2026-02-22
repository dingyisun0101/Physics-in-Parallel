use physics_in_parallel::engines::prelude::*;

#[test]
/// Annotation:
/// - Purpose: Executes `engines_prelude_compiles_for_core_types` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn engines_prelude_compiles_for_core_types() {
    let mut obj = PhysObj::<2>::empty(2);
    obj.set_pos(0, &[1.0, 2.0]).unwrap();
    obj.set_vel(0, &[1.0, 0.0]).unwrap();
    obj.set_pos(1, &[3.0, 2.0]).unwrap();
    obj.set_vel(1, &[0.0, 1.0]).unwrap();
    obj.clear_acc();

    let mut topo = PairTopology::new(2);
    topo.insert(0, 1).unwrap();
    let adj = topo.to_adj_matrix();
    assert_eq!(adj.get(0, 1), 1);

    let mut euler = SemiImplicitEuler;
    euler.step(&mut obj, 0.1).unwrap();

    let mut thermostat = LangevinThermostat::new(1.0, 0.5, 7, false).unwrap();
    thermostat.apply(&mut obj, 0.1).unwrap();

    let temp = TemperatureObserver::default().observe(&obj).unwrap();
    assert!(temp.is_finite());

    let pairs = AllPairs::default().candidates(&obj).unwrap();
    assert_eq!(pairs, vec![(0, 1)]);
}

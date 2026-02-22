use physics_in_parallel::engines::soa::{
    boundary::{Boundary, ClampBox, PeriodicBox, ReflectBox},
    integrator::{ExplicitEuler, Integrator, SemiImplicitEuler},
    neighbors::{AllPairs, NeighborProvider, RadiusPairs},
    observe::{KineticEnergyObserver, MeanReducer, Observer, Reducer, TemperatureObserver},
    phys_obj::PhysObj,
};

#[test]
/// Annotation:
/// - Purpose: Executes `integrators_update_state_with_expected_ordering` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn integrators_update_state_with_expected_ordering() {
    let mut obj_exp = PhysObj::<1>::empty(1);
    obj_exp.set_pos(0, &[0.0]).unwrap();
    obj_exp.set_vel(0, &[1.0]).unwrap();
    obj_exp.set_acc(0, &[2.0]).unwrap();

    let mut explicit = ExplicitEuler;
    explicit.step(&mut obj_exp, 0.5).unwrap();
    assert_eq!(obj_exp.pos_of(0).unwrap(), [0.5].as_slice());
    assert_eq!(obj_exp.vel_of(0).unwrap(), [2.0].as_slice());

    let mut obj_si = PhysObj::<1>::empty(1);
    obj_si.set_pos(0, &[0.0]).unwrap();
    obj_si.set_vel(0, &[1.0]).unwrap();
    obj_si.set_acc(0, &[2.0]).unwrap();

    let mut semi = SemiImplicitEuler;
    semi.step(&mut obj_si, 0.5).unwrap();
    assert_eq!(obj_si.vel_of(0).unwrap(), [2.0].as_slice());
    assert_eq!(obj_si.pos_of(0).unwrap(), [1.0].as_slice());
}

#[test]
/// Annotation:
/// - Purpose: Executes `boundaries_apply_to_positions_and_velocities` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn boundaries_apply_to_positions_and_velocities() {
    let mut obj = PhysObj::<1>::empty(3);
    obj.set_pos(0, &[1.3]).unwrap();
    obj.set_vel(0, &[2.0]).unwrap();
    obj.set_pos(1, &[-0.2]).unwrap();
    obj.set_vel(1, &[3.0]).unwrap();
    obj.set_pos(2, &[1.2]).unwrap();
    obj.set_vel(2, &[4.0]).unwrap();

    PeriodicBox::<1>::new([0.0], [1.0])
        .unwrap()
        .apply_all(&mut obj)
        .unwrap();
    assert!((obj.pos_of(0).unwrap()[0] - 0.3).abs() < 1e-12);
    assert!((obj.pos_of(1).unwrap()[0] - 0.8).abs() < 1e-12);

    ClampBox::<1>::new([0.0], [1.0])
        .unwrap()
        .apply_all(&mut obj)
        .unwrap();
    assert!((obj.pos_of(0).unwrap()[0] - 0.3).abs() < 1e-12);
    assert!((obj.pos_of(1).unwrap()[0] - 0.8).abs() < 1e-12);

    obj.set_pos(2, &[1.2]).unwrap();
    obj.set_vel(2, &[4.0]).unwrap();
    ReflectBox::<1>::new([0.0], [1.0])
        .unwrap()
        .apply_all(&mut obj)
        .unwrap();
    // particle 2 was at 1.2, should reflect to 0.8 and invert velocity.
    assert!((obj.pos_of(2).unwrap()[0] - 0.8).abs() < 1e-12);
    assert!((obj.vel_of(2).unwrap()[0] + 4.0).abs() < 1e-12);
}

#[test]
/// Annotation:
/// - Purpose: Executes `neighbor_providers_filter_pairs` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn neighbor_providers_filter_pairs() {
    let mut obj = PhysObj::<1>::empty(3);
    obj.set_pos(0, &[0.0]).unwrap();
    obj.set_pos(1, &[0.4]).unwrap();
    obj.set_pos(2, &[1.2]).unwrap();

    let all = AllPairs::default().candidates(&obj).unwrap();
    assert_eq!(all, vec![(0, 1), (0, 2), (1, 2)]);

    let radius = RadiusPairs::new(0.5, false).unwrap();
    let near = radius.candidates(&obj).unwrap();
    assert_eq!(near, vec![(0, 1)]);
}

#[test]
/// Annotation:
/// - Purpose: Executes `observers_and_reducers_return_finite_values` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn observers_and_reducers_return_finite_values() {
    let mut obj = PhysObj::<2>::empty(2);
    obj.set_vel(0, &[3.0, 4.0]).unwrap();
    obj.set_vel(1, &[0.0, 2.0]).unwrap();

    let ke = KineticEnergyObserver::default().observe(&obj).unwrap();
    assert!((ke - 14.5).abs() < 1e-12);

    let temp = TemperatureObserver::default().observe(&obj).unwrap();
    assert!((temp - (2.0 * 14.5 / 4.0)).abs() < 1e-12);

    let mean = MeanReducer.reduce(&[1.0, 3.0, 5.0]);
    assert_eq!(mean, 3.0);
}

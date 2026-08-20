use physics_in_parallel::prelude::basic::{
    BoundaryError, ClampBox, ContinuousBoundary, PeriodicBox, ReflectBox,
};

#[test]
fn periodic_and_clamp_update_single_positions() {
    let periodic = PeriodicBox::new(&[0.0], &[1.0]).unwrap();
    let mut r0 = [1.3];
    let mut r1 = [-0.2];
    periodic.apply_position(&mut r0).unwrap();
    periodic.apply_position(&mut r1).unwrap();
    assert!((r0[0] - 0.3).abs() < 1e-12);
    assert!((r1[0] - 0.8).abs() < 1e-12);

    let clamp = ClampBox::new(&[0.0], &[1.0]).unwrap();
    let mut r0 = [1.3];
    let mut r1 = [-0.2];
    clamp.apply_position(&mut r0).unwrap();
    clamp.apply_position(&mut r1).unwrap();
    assert_eq!(r0, [1.0]);
    assert_eq!(r1, [0.0]);
}

#[test]
fn reflect_updates_position_and_velocity_for_large_overshoot() {
    let reflect = ReflectBox::new(&[0.0, 0.0], &[1.0, 1.0]).unwrap();

    let mut r = [1.2, -0.25];
    let mut v = [2.0, -3.0];
    reflect.apply_position_velocity(&mut r, &mut v).unwrap();
    assert_eq!(r, [0.8, 0.25]);
    assert_eq!(v, [-2.0, 3.0]);

    let mut r = [2.2, -1.2];
    let mut v = [5.0, 7.0];
    reflect.apply_position_velocity(&mut r, &mut v).unwrap();
    assert!((r[0] - 0.2).abs() < 1e-12);
    assert!((r[1] - 0.8).abs() < 1e-12);
    assert_eq!(v, [5.0, 7.0]);
}

#[test]
fn bulk_boundary_application_uses_flat_vector_list_layout() {
    let periodic = PeriodicBox::new(&[0.0, 0.0], &[1.0, 1.0]).unwrap();
    let mut positions = vec![1.2, -0.2, 2.4, 0.5];
    periodic.apply_positions(&mut positions).unwrap();
    assert!((positions[0] - 0.2).abs() < 1e-12);
    assert!((positions[1] - 0.8).abs() < 1e-12);
    assert!((positions[2] - 0.4).abs() < 1e-12);
    assert!((positions[3] - 0.5).abs() < 1e-12);

    let reflect = ReflectBox::new(&[0.0], &[1.0]).unwrap();
    let mut positions = vec![1.2, 2.2];
    let mut velocities = vec![3.0, 4.0];
    reflect
        .apply_positions_velocities(&mut positions, &mut velocities)
        .unwrap();
    assert!((positions[0] - 0.8).abs() < 1e-12);
    assert!((positions[1] - 0.2).abs() < 1e-12);
    assert_eq!(velocities, [-3.0, 4.0]);
}

#[test]
fn boundary_rejects_invalid_shapes() {
    assert!(matches!(
        ClampBox::new(&[0.0], &[0.0]).unwrap_err(),
        BoundaryError::InvalidBounds { axis: 0, .. }
    ));

    assert_eq!(
        ReflectBox::new(&[0.0, 0.0], &[1.0]).unwrap_err(),
        BoundaryError::InvalidVectorDimension {
            label: "bounds",
            expected: 2,
            got: 1,
        }
    );

    let periodic = PeriodicBox::new(&[0.0, 0.0], &[1.0, 1.0]).unwrap();
    let mut r = [1.2];
    assert_eq!(
        periodic.apply_position(&mut r).unwrap_err(),
        BoundaryError::InvalidVectorDimension {
            label: "position",
            expected: 2,
            got: 1,
        }
    );

    let mut flat = vec![0.0, 1.0, 2.0];
    assert_eq!(
        periodic.apply_positions(&mut flat).unwrap_err(),
        BoundaryError::InvalidFlatVectorListLength {
            label: "positions",
            dim: 2,
            len: 3,
        }
    );
}

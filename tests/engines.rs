use physics_in_parallel::prelude::basic::Backend;
use physics_in_parallel::prelude::models::{ATTR_R, ParticleStateError, PhysObj, create_template};

#[test]
fn particle_state_exposes_universal_attributes() {
    let mut particles = create_template(2, 2).unwrap();
    particles
        .attribute_mut::<f64>(ATTR_R)
        .unwrap()
        .set_vector(1, &[3.0, 4.0])
        .unwrap();

    let positions = particles.attribute::<f64>(ATTR_R).unwrap();
    assert_eq!(positions.backend(), Backend::Dense);
    assert_eq!(positions.vector(1).unwrap(), vec![3.0, 4.0]);
    assert_eq!(
        particles.attribute_vector::<f64>(ATTR_R, 1).unwrap(),
        vec![3.0, 4.0]
    );
}

#[test]
fn particle_access_reports_model_owned_errors() {
    let particles = create_template(2, 1).unwrap();
    assert!(matches!(
        particles.attribute::<u8>(ATTR_R),
        Err(ParticleStateError::WrongScalarType { .. })
    ));
    assert!(matches!(
        particles.attribute::<f64>("missing"),
        Err(ParticleStateError::MissingAttribute { .. })
    ));
}

#[test]
fn particle_serde_round_trip_uses_the_public_state_type() {
    let particles = create_template(3, 2).unwrap();
    let json = serde_json::to_string(&particles).unwrap();
    let restored: PhysObj = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.num_objects(), Some(2));
    assert_eq!(restored.attribute::<f64>(ATTR_R).unwrap().shape(), [2, 3]);
}

#[test]
fn occupied_neighbors_validate_capacity_and_nonfinite_rebuilds() {
    use physics_in_parallel::advanced::NeighborList;
    assert!(NeighborList::new(&[0.0], &[1.0], f64::MIN_POSITIVE).is_err());
    assert!(NeighborList::new(&[0.0; 3], &[1e10; 3], 1.0).is_err());
    let mut list = NeighborList::new(&[0.0, 0.0], &[1e6, 1e6], 1.0).unwrap();
    assert_eq!(list.num_cells(), 1_000_000_000_000);
    list.rebuild(&[0.0, 0.0, 0.5, 0.0, 9000.0, 8000.0], 3)
        .unwrap();
    assert_eq!(list.collect_pair_candidates(), [(0, 1)]);
    assert!(list.rebuild(&[f64::NAN; 6], 3).is_err());
    assert_eq!(list.collect_pair_candidates(), [(0, 1)]);
    let mut high_rank = NeighborList::new(&[0.0; 20], &[1.0; 20], 2.0).unwrap();
    high_rank.rebuild(&[0.5; 40], 2).unwrap();
    assert_eq!(high_rank.collect_pair_candidates(), [(0, 1)]);
}

#[test]
fn diagonal_kernel_and_object_safe_storage_callback() {
    use physics_in_parallel::advanced::{DiagonalMatrix, RawStorage};
    use physics_in_parallel::math::Tensor;
    let mut diagonal = DiagonalMatrix::<f64>::zeros(100_003, 100_003);
    diagonal.fill(2.0);
    let input = vec![3.0; 100_003];
    let mut output = vec![0.0; input.len()];
    diagonal.mul_vector_into(&input, &mut output).unwrap();
    assert!(output.iter().all(|&value| value == 6.0));
    let mut small = DiagonalMatrix::<f64>::zeros(2, 2);
    small.fill(2.0);
    let mut nonfinite = [0.0; 2];
    small
        .mul_vector_into(&[f64::INFINITY, 1.0], &mut nonfinite)
        .unwrap();
    assert!(nonfinite[0].is_infinite() && nonfinite[1].is_nan());
    let tensor =
        Tensor::from_entries(&[1_000_000], Backend::Sparse, vec![(vec![42], 5.0)]).unwrap();
    let raw: &dyn RawStorage<f64> = &tensor;
    let mut entries = Vec::new();
    raw.for_each_stored_entry(&mut |index, value| entries.push((index, value)));
    assert_eq!(entries, [(42, 5.0)]);
}

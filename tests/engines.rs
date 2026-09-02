use physics_in_parallel::prelude::basic::StorageKind;
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
    assert_eq!(positions.storage_kind(), StorageKind::Dense);
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

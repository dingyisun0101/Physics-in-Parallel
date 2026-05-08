use std::collections::BTreeSet;

use physics_in_parallel::models::particles::attrs::{ATTR_R, ParticleSelection, set_alive};
use physics_in_parallel::models::particles::create_state::create_template;
use physics_in_parallel::models::particles::interactions::{
    ParticleNeighborList, ParticleNeighborListError,
};

fn pair_set(pairs: Vec<(usize, usize)>) -> BTreeSet<(usize, usize)> {
    pairs.into_iter().collect()
}

#[test]
fn particle_neighbor_list_collects_cutoff_filtered_pairs() {
    let mut objects = create_template(2, 4).unwrap();
    objects
        .core
        .set_vector_of::<f64>(ATTR_R, 0, &[0.0, 0.0])
        .unwrap();
    objects
        .core
        .set_vector_of::<f64>(ATTR_R, 1, &[0.5, 0.0])
        .unwrap();
    objects
        .core
        .set_vector_of::<f64>(ATTR_R, 2, &[1.4, 0.0])
        .unwrap();
    objects
        .core
        .set_vector_of::<f64>(ATTR_R, 3, &[4.0, 0.0])
        .unwrap();

    let mut neighbors = ParticleNeighborList::from_bounds(&[0.0, 0.0], &[5.0, 1.0], 1.0).unwrap();
    assert_eq!(neighbors.dim(), 2);
    assert_eq!(neighbors.cutoff(), 1.0);

    neighbors.rebuild(&objects).unwrap();
    assert_eq!(neighbors.candidates().num_objects(), 4);

    let pairs = pair_set(
        neighbors
            .collect_pairs(&objects, ParticleSelection::AliveOnly)
            .unwrap(),
    );
    assert_eq!(pairs, BTreeSet::from([(0, 1), (1, 2)]));
}

#[test]
fn particle_neighbor_list_alive_mask_filters_dead_particles() {
    let mut objects = create_template(2, 3).unwrap();

    objects
        .core
        .set_vector_of::<f64>(ATTR_R, 0, &[0.0, 0.0])
        .unwrap();
    objects
        .core
        .set_vector_of::<f64>(ATTR_R, 1, &[0.5, 0.0])
        .unwrap();
    objects
        .core
        .set_vector_of::<f64>(ATTR_R, 2, &[0.8, 0.0])
        .unwrap();

    set_alive(&mut objects, 0, true).unwrap();
    set_alive(&mut objects, 1, false).unwrap();
    set_alive(&mut objects, 2, true).unwrap();

    let mut neighbors = ParticleNeighborList::from_bounds(&[0.0, 0.0], &[2.0, 1.0], 1.0).unwrap();
    neighbors.rebuild(&objects).unwrap();

    let all_slots = pair_set(
        neighbors
            .collect_pairs(&objects, ParticleSelection::All)
            .unwrap(),
    );
    assert_eq!(all_slots, BTreeSet::from([(0, 1), (0, 2), (1, 2)]));

    let alive_only = pair_set(
        neighbors
            .collect_pairs(&objects, ParticleSelection::AliveOnly)
            .unwrap(),
    );
    assert_eq!(alive_only, BTreeSet::from([(0, 2)]));
}

#[test]
fn particle_neighbor_list_rejects_position_dimension_mismatch() {
    let objects = create_template(3, 2).unwrap();
    let mut neighbors = ParticleNeighborList::from_bounds(&[0.0, 0.0], &[2.0, 2.0], 1.0).unwrap();

    let err = neighbors.rebuild(&objects).unwrap_err();

    assert_eq!(
        err,
        ParticleNeighborListError::InvalidAttrShape {
            label: ATTR_R,
            expected_dim: 2,
            got_dim: 3,
        }
    );
}

use std::collections::BTreeSet;

use physics_in_parallel::prelude::advanced::{NeighborList, NeighborListError};

fn pair_set(pairs: Vec<(usize, usize)>) -> BTreeSet<(usize, usize)> {
    pairs.into_iter().collect()
}

#[test]
fn neighbor_list_reports_configuration_and_rebuild_state() {
    let mut list = NeighborList::new(&[0.0, 0.0], &[4.0, 2.0], 1.0).unwrap();

    assert_eq!(list.dim(), 2);
    assert_eq!(list.min(), &[0.0, 0.0]);
    assert_eq!(list.max(), &[4.0, 2.0]);
    assert_eq!(list.cell_width(), 1.0);
    assert_eq!(list.cells_per_axis(), &[4, 2]);
    assert_eq!(list.num_cells(), 8);
    assert_eq!(list.num_objects(), 0);

    let positions = [0.2, 0.2, 1.2, 0.2, 3.6, 1.7];
    list.rebuild(&positions, 3).unwrap();
    assert_eq!(list.num_objects(), 3);

    list.clear();
    assert_eq!(list.num_objects(), 0);
    assert_eq!(list.num_cells(), 8);
}

#[test]
fn neighbor_list_rejects_invalid_inputs() {
    assert_eq!(
        NeighborList::new(&[0.0], &[1.0], 0.0).unwrap_err(),
        NeighborListError::InvalidCellWidth { cell_width: 0.0 }
    );

    assert_eq!(
        NeighborList::new(&[0.0], &[0.0], 1.0).unwrap_err(),
        NeighborListError::InvalidBounds {
            axis: 0,
            min: 0.0,
            max: 0.0,
        }
    );

    let mut list = NeighborList::new(&[0.0, 0.0], &[1.0, 1.0], 0.5).unwrap();
    assert_eq!(
        list.rebuild(&[0.0, 0.0, 0.5], 2).unwrap_err(),
        NeighborListError::InvalidPositionShape {
            expected_len: 4,
            got_len: 3,
        }
    );
}

#[test]
fn neighbor_list_emits_unique_sorted_same_and_adjacent_cell_candidates() {
    let mut list = NeighborList::new(&[0.0], &[5.0], 1.0).unwrap();
    let positions = [
        0.1, // object 0, cell 0
        0.2, // object 1, cell 0
        1.1, // object 2, cell 1
        3.1, // object 3, cell 3
    ];
    list.rebuild(&positions, 4).unwrap();

    let pairs = pair_set(list.collect_pair_candidates());
    assert_eq!(pairs, BTreeSet::from([(0, 1), (0, 2), (1, 2)]));
}

#[test]
fn neighbor_list_clips_out_of_bound_positions_into_valid_cells() {
    let mut list = NeighborList::new(&[0.0], &[2.0], 1.0).unwrap();
    let positions = [
        -10.0, // clips to first cell
        0.25,  // first cell
        99.0,  // clips to last cell
        1.25,  // last cell
    ];
    list.rebuild(&positions, 4).unwrap();

    let pairs = pair_set(list.collect_pair_candidates());
    assert_eq!(
        pairs,
        BTreeSet::from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    );
}

#[test]
fn neighbor_list_empty_rebuild_has_no_pairs() {
    let mut list = NeighborList::new(&[0.0, 0.0], &[1.0, 1.0], 0.5).unwrap();

    list.rebuild(&[], 0).unwrap();

    assert_eq!(list.num_objects(), 0);
    assert!(list.collect_pair_candidates().is_empty());
}

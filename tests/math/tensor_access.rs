use physics_in_parallel::prelude::advanced::{Dense, RowMajorLayout, Sparse};
use physics_in_parallel::prelude::basic::{
    DEFAULT_PARALLEL_PARTITIONS, ParallelismError, Tensor, parallel_partitions,
    set_parallel_partitions,
};

#[test]
fn process_parallel_partition_policy_has_a_safe_default() {
    assert_eq!(DEFAULT_PARALLEL_PARTITIONS, 8);
    assert_eq!(
        set_parallel_partitions(0),
        Err(ParallelismError::ZeroPartitions)
    );
    assert_eq!(parallel_partitions(), DEFAULT_PARALLEL_PARTITIONS);
}

#[test]
fn row_major_layout_round_trips_flat_and_wrapped_coordinates() {
    let layout = RowMajorLayout::try_new(&[2, 3, 5]).unwrap();
    assert_eq!(layout.size(), 30);
    assert_eq!(layout.strides(), [15, 5, 1]);

    let mut coordinate = [0; 3];
    for flat in 0..layout.size() {
        layout.coordinate_into(flat, &mut coordinate);
        assert_eq!(layout.index_wrapped(&coordinate), flat);
        assert_eq!(layout.coordinate(flat).unwrap(), coordinate);
    }
    assert_eq!(layout.index_wrapped(&[-1, -1, -1]), 29);
    assert_eq!(layout.index_wrapped(&[2, 3, 5]), 0);
    assert_eq!(layout.coordinate(layout.size()), None);
}

#[test]
fn dense_get_set_and_periodic_indexing_are_consistent() {
    let mut t = Tensor::<i64, Dense>::empty(&[2, 3]);
    t.set(&[0, 0], 10);
    t.set(&[0, 1], 11);
    t.set(&[0, 2], 12);
    t.set(&[1, 0], 20);
    t.set(&[1, 1], 21);
    t.set(&[1, 2], 22);

    assert_eq!(t.get(&[0, 0]), 10);
    assert_eq!(t.get(&[1, 2]), 22);
    assert_eq!(t.get(&[-1, -1]), 22);
    assert_eq!(t.get(&[2, 3]), 10);
    assert_eq!(t.get(&[-1, -1]), t.get(&[1, 2]));

    assert_eq!(t.flat_index(&[-1, -1]), 5);
    assert_eq!(t.get_flat(-1), 22);
    *t.get_flat_mut(6) = 99;
    assert_eq!(t.get(&[0, 0]), 99);
    t.set_flat(-2, 77);
    assert_eq!(t.get(&[1, 1]), 77);
}

#[test]
fn tensor_layout_rejects_shapes_outside_signed_index_space() {
    let error = RowMajorLayout::try_new(&[isize::MAX as usize, 2]).unwrap_err();
    assert!(matches!(
        error,
        physics_in_parallel::prelude::basic::TensorError::IndexSpaceOverflow { .. }
            | physics_in_parallel::prelude::basic::TensorError::ShapeProductOverflow { .. }
    ));
}

#[test]
fn dense_mutable_reference_updates_target_slot() {
    let mut t = Tensor::<i64, Dense>::empty(&[2, 2]);
    t.set(&[0, 0], 1);
    t.set(&[0, 1], 2);
    t.set(&[1, 0], 3);
    t.set(&[1, 1], 4);

    {
        let slot = t.get_mut(&[-1, -1]);
        *slot = 99;
    }

    assert_eq!(t.get(&[1, 1]), 99);
    assert_eq!(t.get(&[-1, -1]), 99);

    t.set(&[2, 2], -5);
    assert_eq!(t.get(&[0, 0]), -5);
}

#[test]
fn sparse_get_set_zero_pruning_and_wrapping_are_consistent() {
    let mut t = Tensor::<i64, Sparse>::empty(&[2, 3]);
    t.set(&[0, 0], 10);
    t.set(&[1, 2], 22);

    assert_eq!(t.nnz(), 2);
    assert_eq!(t.get(&[0, 1]), 0);
    assert_eq!(t.get(&[-1, -1]), 22);
    assert_eq!(t.get(&[2, 3]), 10);

    t.set(&[-1, -1], 0);
    assert_eq!(t.get(&[1, 2]), 0);
    assert_eq!(t.nnz(), 1);

    t.set(&[2, 3], 33);
    assert_eq!(t.get(&[0, 0]), 33);
}

#[test]
fn sparse_mutable_reference_materializes_and_prune_removes_zero() {
    let mut t = Tensor::<i64, Sparse>::empty(&[2, 2]);
    assert_eq!(t.nnz(), 0);

    {
        let slot = t.get_mut(&[1, 1]);
        assert_eq!(*slot, 0);
        *slot = 7;
    }

    assert_eq!(t.nnz(), 1);
    assert_eq!(t.get(&[1, 1]), 7);

    {
        let slot = t.get_mut(&[-1, -1]);
        *slot = 0;
    }

    assert_eq!(t.get(&[1, 1]), 0);
    assert_eq!(t.nnz(), 1);
    t.set(&[1, 1], 0);
    assert_eq!(t.nnz(), 0);
}

#[test]
fn generic_public_access_updates_backend_without_exposing_storage() {
    let mut dense = Tensor::<i64, Dense>::empty(&[1, 2]);
    dense.set(&[0, 0], 5);
    dense.set(&[0, 1], 6);
    assert_eq!(dense.shape(), &[1, 2]);
    *dense.get_mut(&[0, 1]) = 60;
    assert_eq!(dense.get(&[0, 1]), 60);

    let mut sparse = Tensor::<i64, Sparse>::empty(&[1, 2]);
    sparse.set(&[0, 1], 9);
    assert_eq!(sparse.shape(), &[1, 2]);
    assert_eq!(sparse.get(&[0, 1]), 9);
    assert_eq!(sparse.nnz(), 1);
}

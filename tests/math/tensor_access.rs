use physics_in_parallel::prelude::advanced::{Dense, RowMajorLayout, Sparse};
use physics_in_parallel::prelude::basic::{ComputePool, ComputePoolError, Tensor, with_threads};

#[test]
fn explicit_compute_pools_are_fixed_size_and_opt_in() {
    assert!(matches!(
        ComputePool::new(0),
        Err(ComputePoolError::ZeroThreads)
    ));

    let pool = ComputePool::new(2).unwrap();
    assert_eq!(pool.threads(), 2);
    assert_eq!(pool.install(rayon::current_num_threads), 2);
    assert_eq!(with_threads(3, rayon::current_num_threads).unwrap(), 3);
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
fn dense_contiguous_ownership_and_allocation_reuse_are_explicit() {
    let mut target = Tensor::<i64>::try_from_vec(&[2, 3], vec![0; 6]).unwrap();
    let left = Tensor::<i64>::try_from_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let right = Tensor::<i64>::try_from_vec(&[2, 3], vec![6, 5, 4, 3, 2, 1]).unwrap();

    target.zip_from(&left, &right, |a, b| a + b).unwrap();
    assert_eq!(target.as_slice(), &[7; 6]);
    target.as_mut_slice()[0] = 9;
    assert_eq!(target.get_flat(0), 9);

    target.copy_from(&left).unwrap();
    assert_eq!(target, left);
    assert_eq!(target.sum_serial(), 21);
}

#[test]
fn dense_fallible_construction_and_copy_report_layout_errors() {
    assert!(matches!(
        Tensor::<f64>::try_from_vec(&[2, 2], vec![1.0, 2.0, 3.0]),
        Err(
            physics_in_parallel::prelude::basic::TensorError::DataLengthMismatch {
                expected: 4,
                actual: 3
            }
        )
    ));

    let mut target = Tensor::<f64>::zeros(&[2, 2]);
    let source = Tensor::<f64>::zeros(&[4]);
    assert!(matches!(
        target.copy_from(&source),
        Err(physics_in_parallel::prelude::basic::TensorError::ShapeMismatch { .. })
    ));
}

#[test]
fn dense_adaptive_chunk_traversal_visits_each_chunk_once() {
    let mut values = Tensor::<usize>::zeros(&[5_000, 4]);
    values.for_each_chunk_mut(4, |row, chunk| {
        for (column, value) in chunk.iter_mut().enumerate() {
            *value = row * 4 + column;
        }
    });
    assert_eq!(values.as_slice(), (0..20_000).collect::<Vec<_>>());
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

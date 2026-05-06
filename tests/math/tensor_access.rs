use physics_in_parallel::math::tensor::{Dense, Sparse, Tensor, TensorTrait};

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
    assert_eq!(t.storage().index(&[-1, -1]), t.storage().index(&[1, 2]));
}

#[test]
fn dense_mutable_reference_updates_target_slot() {
    let mut t = Tensor::<i64, Dense>::empty(&[2, 2]);
    t.set(&[0, 0], 1);
    t.set(&[0, 1], 2);
    t.set(&[1, 0], 3);
    t.set(&[1, 1], 4);

    {
        let slot = t.storage_mut().get_mut(&[-1, -1]);
        *slot = 99;
    }

    assert_eq!(t.get(&[1, 1]), 99);
    assert_eq!(t.get(&[-1, -1]), 99);

    {
        let slot = t.storage_mut().get_by_idx_mut(4);
        *slot = -5;
    }

    assert_eq!(t.get(&[0, 0]), -5);
    assert_eq!(*t.storage().get_by_idx(4), -5);
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

    t.storage_mut().set_by_flat(6, 33);
    assert_eq!(t.get(&[0, 0]), 33);
    assert_eq!(t.storage().get_by_flat(6), 33);
}

#[test]
fn sparse_mutable_reference_materializes_and_prune_removes_zero() {
    let mut t = Tensor::<i64, Sparse>::empty(&[2, 2]);
    assert_eq!(t.nnz(), 0);

    {
        let slot = t.storage_mut().get_mut(&[1, 1]);
        assert_eq!(*slot, 0);
        *slot = 7;
    }

    assert_eq!(t.nnz(), 1);
    assert_eq!(t.get(&[1, 1]), 7);

    {
        let slot = t.storage_mut().get_mut(&[-1, -1]);
        *slot = 0;
    }

    assert_eq!(t.get(&[1, 1]), 0);
    assert_eq!(t.nnz(), 1);
    t.storage_mut().prune_zeros();
    assert_eq!(t.nnz(), 0);
}

#[test]
fn generic_storage_references_expose_backend_without_copying() {
    let mut dense = Tensor::<i64, Dense>::empty(&[1, 2]);
    dense.set(&[0, 0], 5);
    dense.set(&[0, 1], 6);
    assert_eq!(dense.storage().shape(), &[1, 2]);
    dense.storage_mut().set(&[0, 1], 60);
    assert_eq!(dense.get(&[0, 1]), 60);

    let mut sparse = Tensor::<i64, Sparse>::empty(&[1, 2]);
    sparse.storage_mut().set(&[0, 1], 9);
    assert_eq!(sparse.storage().shape(), &[1, 2]);
    assert_eq!(sparse.get(&[0, 1]), 9);
    assert_eq!(sparse.nnz(), 1);
}

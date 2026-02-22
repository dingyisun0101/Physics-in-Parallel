use physics_in_parallel::math::tensor::core::{
    dense::Tensor as DenseTensor,
    sparse::Tensor as SparseTensor,
    tensor_trait::TensorTrait,
};

#[test]
/// Annotation:
/// - Purpose: Executes `sparse_public_surface_basics` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn sparse_public_surface_basics() {
    let mut s = SparseTensor::<f64>::from_triplets(
        vec![2, 3],
        vec![(vec![0, 1], 2.0), (vec![1, 2], 3.0)],
    );

    assert_eq!(s.rank(), 2);
    assert_eq!(s.shape(), &[2, 3]);
    assert_eq!(s.len_dense(), 6);
    assert_eq!(s.nnz(), 2);
    assert!(!s.is_empty());

    assert_eq!(s.index(&[0, 1]), 1);
    assert_eq!(s.get_opt(&[0, 1]).copied(), Some(2.0));
    assert_eq!(s.get(&[0, 2]), 0.0);

    *s.get_mut_or_insert_zero(&[0, 0]) = 5.0;
    s.prune_zeros();
    s.set(&[0, 0], 0.0);

    s.add_assign_at(&[0, 1], 1.5);
    assert_eq!(s.get(&[0, 1]), 3.5);

    s.set_by_flat(6, 7.0);
    assert_eq!(s.get_by_flat(0), 7.0);

    let dense = s.to_dense();
    let back = SparseTensor::<f64>::from_dense(&dense);
    assert_eq!(back.shape(), s.shape());

    let cast_i = s.cast_to::<i64>();
    assert_eq!(cast_i.get(&[0, 1]), 3);
    let try_cast_f32 = s.try_cast_to::<f32>().expect("f64->f32 cast should succeed");
    assert_eq!(try_cast_f32.shape(), s.shape());

    let mut t = <SparseTensor<f64> as TensorTrait<f64>>::empty(&[2, 2]);
    t.set(&[0, 0], 2.0);
    t.par_fill(1.0);
    t.par_map_in_place(|x| x + 1.0);
    t.par_zip_with_inplace(&back, |a, b| a + b);
    let _ = <SparseTensor<f64> as TensorTrait<f64>>::cast_to::<i64>(&t);
}

#[test]
/// Annotation:
/// - Purpose: Executes `sparse_integer_bitand_and_scalar_ops` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn sparse_integer_bitand_and_scalar_ops() {
    let a = SparseTensor::<i64>::from_triplets(vec![2], vec![(vec![0], 3), (vec![1], 4)]);
    let b = SparseTensor::<i64>::from_triplets(vec![2], vec![(vec![0], 1), (vec![1], 2)]);

    let c = a.clone() & b.clone();
    assert_eq!(c.get(&[0]), 1);
    assert_eq!(c.get(&[1]), 0);

    let d = a.clone() + b.clone();
    let e = a.clone() - b.clone();
    let f = a.clone() * b.clone();
    let g = a.clone() / b.clone();
    assert_eq!(d.get(&[0]), 4);
    assert_eq!(e.get(&[1]), 2);
    assert_eq!(f.get(&[0]), 3);
    assert_eq!(g.get(&[1]), 2);

    let h = a.clone() + 1;
    let i = a.clone() - 1;
    let j = a.clone() * 2;
    let k = a.clone() / 2;
    assert_eq!(h.get(&[0]), 4);
    assert_eq!(i.get(&[1]), 3);
    assert_eq!(j.get(&[0]), 6);
    assert_eq!(k.get(&[1]), 2);
}

#[test]
/// Annotation:
/// - Purpose: Executes `sparse_dense_roundtrip_on_trait_entry_points` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn sparse_dense_roundtrip_on_trait_entry_points() {
    let mut d = <DenseTensor<f64> as TensorTrait<f64>>::empty(&[2, 2]);
    d.set(&[0, 1], 5.0);
    d.set(&[1, 0], 7.0);

    let s = SparseTensor::<f64>::from_dense(&d);
    assert_eq!(s.nnz(), 2);

    let d2 = s.to_dense();
    assert_eq!(d2.get(&[0, 1]), 5.0);
    assert_eq!(d2.get(&[1, 0]), 7.0);
}

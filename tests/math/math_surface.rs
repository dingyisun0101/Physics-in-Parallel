use ndarray::{arr2, ArrayD, IxDyn};

use physics_in_parallel::math::{
    ndarray_convert::NdarrayConvert,
    prelude::*,
    tensor::{
        core::{
            dense::Tensor as DenseTensorStorage,
            dense_rand::{RandType, TensorRandFiller},
            sparse::Tensor as SparseTensorStorage,
            tensor_trait::TensorTrait,
        },
        rank_2::{
            dense::Tensor2D,
            matrix::{dense::Matrix, matrix_trait::MatrixTrait},
            vector_list::VectorList,
            vector_list_rand::{HaarVectors, NNVectors},
        },
    },
};

#[test]
#[allow(deprecated)]
/// Annotation:
/// - Purpose: Executes `dense_tensor_public_surface` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn dense_tensor_public_surface() {
    let mut t = <DenseTensorStorage<f64> as TensorTrait<f64>>::empty(&[2, 3]);
    assert_eq!(t.len(), 6);
    assert!(!t.is_empty());

    t.set_by_idx(0, 1.0);
    *t.get_by_idx_mut(1) = 2.0;
    assert_eq!(*t.get_by_idx(0), 1.0);
    assert_eq!(*t.get_by_idx(1), 2.0);

    assert_eq!(t.index(&[0, 2]), 2);
    assert_eq!(t.get(&[0, 2]), 0.0);
    *t.get_mut(&[1, 2]) = 3.0;
    t.set(&[0, 2], 4.0);
    assert_eq!(t.get_sum(), 10.0);

    t.par_fill(1.0);
    t.par_map_in_place(|x| x + 1.0);
    let t2 = <DenseTensorStorage<f64> as TensorTrait<f64>>::empty(&[2, 3]);
    t.par_zip_with_inplace(&t2, |a, b| a + b);
    let cast_i = <DenseTensorStorage<f64> as TensorTrait<f64>>::cast_to::<i64>(&t);
    assert_eq!(cast_i.shape.as_slice(), &[2, 3]);
    t.print();

    let s = t.to_string();
    let parsed = DenseTensorStorage::<f64>::from_string(&s);
    assert_eq!(parsed.shape, t.shape);

    let try_cast: DenseTensorStorage<f32> = t.try_cast_to().expect("dense cast");
    assert_eq!(try_cast.shape, t.shape);

    let sp = t.to_sparse();
    let back = DenseTensorStorage::<f64>::from_sparse(&sp);
    assert_eq!(back.shape, t.shape);

    let arr = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let from_a = DenseTensorStorage::<f64>::from_ndarray(&arr);
    let _from_a_depr = DenseTensorStorage::<f64>::from_ndarry(&arr);
    let to_a = from_a.to_ndarray();
    assert_eq!(to_a, arr);

    let mut alias = from_a.clone();
    alias.par_map_inplace(|x| x * 2.0);
    alias.print_2d();

    // arithmetic operators
    let mut op_l = from_a.clone();
    let op_r = from_a.clone();
    let _ = &op_l + &op_r;
    let _ = &op_l - &op_r;
    let _ = &op_l * &op_r;
    let _ = &op_l / &op_r;
    op_l += &op_r;
    op_l -= &op_r;
    op_l *= &op_r;
    op_l /= &op_r;
    let _ = &op_l + 1.0;
    let _ = &op_l - 1.0;
    let _ = &op_l * 2.0;
    let _ = &op_l / 2.0;
    op_l += 1.0;
    op_l -= 1.0;
    op_l *= 2.0;
    op_l /= 2.0;
}

#[test]
/// Annotation:
/// - Purpose: Executes `sparse_tensor_public_surface` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn sparse_tensor_public_surface() {
    let mut s = SparseTensorStorage::<f64>::from_triplets(
        vec![2, 3],
        vec![(vec![0, 1], 2.0), (vec![1, 2], 3.0)],
    );
    assert_eq!(s.len_dense(), 6);
    assert_eq!(s.rank(), 2);
    assert_eq!(s.shape(), [2, 3]);
    assert_eq!(s.nnz(), 2);
    assert!(!s.is_empty());

    let k = s.index(&[0, 1]);
    assert_eq!(k, 1);
    assert_eq!(s.get_opt(&[0, 1]).copied(), Some(2.0));
    assert_eq!(s.get(&[0, 2]), 0.0);

    *s.get_mut_or_insert_zero(&[0, 0]) = 5.0;
    s.prune_zeros();
    s.set(&[0, 0], 0.0);
    s.add_assign_at(&[0, 1], 1.0);
    assert_eq!(s.get(&[0, 1]), 3.0);
    assert!(s.iter().count() >= 1);

    s.set_by_flat_unchecked(0, 7.0);
    s.set_by_flat(6, 8.0); // wraps to 0
    assert_eq!(s.get_by_flat(0), 8.0);
    assert_eq!(s.get_by_flat(6), 8.0);

    let cast_i: SparseTensorStorage<i64> = s.try_cast_to().expect("sparse cast");
    let _cast_f: SparseTensorStorage<f32> = cast_i.cast_to();

    let d = s.to_dense();
    let back = SparseTensorStorage::<f64>::from_dense(&d);
    assert_eq!(back.shape(), s.shape());

    let arr = s.to_ndarray();
    let s2 = SparseTensorStorage::<f64>::from_ndarray(&arr);
    assert_eq!(s2.shape(), s.shape());

    // TensorTrait methods
    s.par_fill(1.0);
    s.par_map_in_place(|x| x + 1.0);
    s.par_zip_with_inplace(&s2, |a, b| a + b);
    let _ = <SparseTensorStorage<f64> as TensorTrait<f64>>::cast_to::<i64>(&s);
    s.print();

    // ops
    let mut a = s.clone();
    let b = s2.clone();
    let _ = a.clone() + b.clone();
    let _ = a.clone() - b.clone();
    let _ = a.clone() * b.clone();
    let _ = a.clone() / b.clone();
    let _ = a.clone() + 1.0;
    let _ = a.clone() - 1.0;
    let _ = a.clone() * 2.0;
    let _ = a.clone() / 2.0;
    a = a + 1.0;
    assert!(a.nnz() >= 1);

    // bitand is defined for integer-like scalar backends
    let si1 = SparseTensorStorage::<i64>::from_triplets(vec![2], vec![(vec![0], 3_i64)]);
    let si2 = SparseTensorStorage::<i64>::from_triplets(vec![2], vec![(vec![0], 1_i64)]);
    let si3 = si1 & si2;
    assert_eq!(si3.get(&[0]), 1);
}

#[test]
/// Annotation:
/// - Purpose: Executes `dense_rand_public_surface` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn dense_rand_public_surface() {
    let mut t_f = <DenseTensorStorage<f64> as TensorTrait<f64>>::empty(&[4]);
    let mut t_i = <DenseTensorStorage<i64> as TensorTrait<i64>>::empty(&[4]);
    let mut t_u = <DenseTensorStorage<usize> as TensorTrait<usize>>::empty(&[4]);
    let mut t_s = <DenseTensorStorage<isize> as TensorTrait<isize>>::empty(&[4]);

    let mut rf = TensorRandFiller::new(
        RandType::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Some(2),
    );
    rf.refresh(&mut t_f);
    assert!(matches!(rf.kind(), RandType::Uniform { .. }));
    rf.set_kind(RandType::Normal {
        mean: 0.0,
        std: 1.0,
    });
    rf.refresh(&mut t_f);

    let mut ri =
        TensorRandFiller::new_with_seed(RandType::UniformInt { low: 1, high: 3 }, Some(2), 7);
    ri.refresh(&mut t_i);
    ri.set_kind(RandType::Bernoulli { p: 0.5 });
    ri.refresh(&mut t_i);

    let mut ru = TensorRandFiller::new(RandType::UniformInt { low: 0, high: 2 }, Some(2));
    ru.refresh(&mut t_u);

    let mut rs = TensorRandFiller::new(RandType::UniformInt { low: -2, high: 2 }, Some(2));
    rs.refresh(&mut t_s);
}

#[test]
/// Annotation:
/// - Purpose: Executes `unified_tensor_public_surface` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn unified_tensor_public_surface() {
    let dense_storage = <DenseTensorStorage<f64> as TensorTrait<f64>>::empty(&[2, 2]);
    let mut td = Tensor::<f64, DenseBackend>::from_storage(dense_storage);
    assert_eq!(td.shape(), [2, 2]);
    td.storage_mut().set(&[0, 0], 1.0);
    assert_eq!(td.storage().get(&[0, 0]), 1.0);
    td.set(&[1, 1], 2.0);
    assert_eq!(td.get_sum(), 3.0);
    td.par_fill(2.0);
    td.par_map_in_place(|x| x + 1.0);
    let td2 = Tensor::<f64, DenseBackend>::empty(&[2, 2]);
    td.par_zip_with_inplace(&td2, |a, b| a + b);
    td.print();

    let _cast = td.cast_to::<i64>();
    let _try_cast = td.try_cast_to::<f32>().expect("unified dense cast");
    let ts = td.to_sparse();
    let _back = Tensor::<f64, DenseBackend>::from_sparse(&ts);
    let arr = td.to_ndarray();
    let _from_arr = Tensor::<f64, DenseBackend>::from_ndarray(&arr);

    let sparse_storage = <SparseTensorStorage<f64> as TensorTrait<f64>>::empty(&[2, 2]);
    let mut ts2 = Tensor::<f64, SparseBackend>::from_storage(sparse_storage);
    ts2.set(&[0, 1], 4.0);
    assert_eq!(ts2.nnz(), 1);
    assert_eq!(ts2.len_dense(), 4);
    let _scast = ts2.cast_to::<i64>();
    let _stry = ts2.try_cast_to::<f32>().expect("unified sparse cast");
    let td3 = ts2.to_dense();
    let _from_dense = Tensor::<f64, SparseBackend>::from_dense(&td3);
    let arrs = ts2.to_ndarray();
    let _from_arrs = Tensor::<f64, SparseBackend>::from_ndarray(&arrs);
    let _inner = ts2.into_storage();
}

#[test]
#[allow(deprecated)]
/// Annotation:
/// - Purpose: Executes `tensor2d_matrix_vector_list_public_surface` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn tensor2d_matrix_vector_list_public_surface() {
    let mut t2 = Tensor2D::<f64>::empty(2, 3);
    assert_eq!(t2.rows(), 2);
    assert_eq!(t2.cols(), 3);
    assert_eq!(t2.shape(), [2, 3]);
    t2.set(0, 1, 2.0);
    assert_eq!(t2.get(0, 1), 2.0);
    assert_eq!(t2.backend().shape.as_slice(), &[2, 3]);
    t2.backend_mut().set(&[1, 2], 3.0);
    assert_eq!(t2.data().len(), 6);
    t2.data_mut()[0] = 1.0;
    assert_eq!(t2.row_view(0), [1.0, 2.0, 0.0]);
    t2.row_view_mut(1)[0] = 5.0;
    assert_eq!(t2.col_to_vec(0), vec![1.0, 5.0]);
    t2.set_col_from_slice(2, &[7.0, 8.0]);
    let _t2i = t2.cast_to::<i64>();
    let arr2d = t2.to_ndarray();
    let _t2b = Tensor2D::<f64>::from_ndarray(&arr2d);
    t2.print();
    t2.to_string();
    let _t2_from_backend = Tensor2D::<f64>::from_backend(t2.backend().clone(), 2, 3);

    let mut m = Matrix::<f64>::empty(2, 2);
    m.set(0, 0, 1.0);
    m.set(0, 1, 2.0);
    m.set(1, 0, 3.0);
    m.set(1, 1, 4.0);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 2);
    assert_eq!(m.shape(), [2, 2]);
    let _mb = m.backend();
    let _mbm = m.backend_mut();
    let _db = m.dense_backend();
    let _dbm = m.dense_backend_mut();
    let _mfrom = Matrix::<f64>::from_tensor(m.dense_backend().clone());
    let _mfromb = Matrix::<f64>::from_backend(t2.clone().cast_to::<f64>(), 2, 3);
    let _mcast = m.cast_to::<i64>();
    let ma = m.to_ndarray();
    let _mfroma = Matrix::<f64>::from_ndarray(&ma);
    let _mfroma2 = Matrix::<f64>::from_ndarry(&ma);
    assert_eq!(m.get(1, 1), 4.0);

    assert_eq!(&*m.get_row_ref(0), [1.0, 2.0]);
    assert_eq!(&*m.get_col_ref(1), [2.0, 4.0]);
    m.get_row_ref_mut(0)[0] = 10.0;
    {
        let mut c = m.get_col_ref_mut(0);
        c[1] = 30.0;
    }
    assert_eq!(m.get(1, 0), 30.0);

    MatrixTrait::set_row_from_slice(&mut m, 1, &[6.0, 8.0]);
    MatrixTrait::set_col_from_slice(&mut m, 0, &[9.0, 7.0]);
    assert_eq!(MatrixTrait::row_view(&m, 0), [9.0, 2.0]);
    MatrixTrait::row_view_mut(&mut m, 0)[1] = 11.0;
    MatrixTrait::print(&m);
    MatrixTrait::to_string(&m);
    MatrixTrait::transpose(&mut m);
    MatrixTrait::clamp(&mut m, 0.0, 10.0);
    MatrixTrait::normalize(&mut m);
    MatrixTrait::normalize_by_max(&mut m);

    let _ = &m + &m;
    let _ = &m - &m;
    let _ = &m * &m;
    let _ = &m / &m;
    let mut m2 = m.clone();
    m2 += &m;
    m2 -= &m;
    m2 *= &m;
    m2 /= &m;
    let _ = &m + 1.0;
    let _ = &m - 1.0;
    let _ = &m * 2.0;
    let _ = &m / 2.0;
    m2 += 1.0;
    m2 -= 1.0;
    m2 *= 2.0;
    m2 /= 2.0;

    let mut vl = VectorList::<f64>::empty(3, 2);
    let _vl_from_t2 = VectorList::<f64>::from_tensor2d(Tensor2D::<f64>::empty(2, 3));
    let _into_t2 = vl.clone().into_tensor2d();
    let _vt = vl.as_tensor();
    let _vtm = vl.as_tensor_mut();
    vl.print();
    let _vl_cast = vl.cast_to::<i64>();
    let arr_v = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    vl = VectorList::<f64>::from_ndarray(&arr_v);
    let _arr_back = vl.to_ndarray();
    assert_eq!(vl.get(1, 2), 6.0);
    vl.set(1, 2, 7.0);
    assert_eq!(vl.get_vector(1), [2.0, 4.0, 7.0]);
    vl.get_vector_mut(0)[0] = 9.0;
    assert_eq!(vl.get_axis(0), vec![9.0, 2.0]);
    vl.scale_vectors_by_list(&[2.0, 3.0]);
    vl.set_vector_from_slice(0, &[1.0, 1.0, 1.0]);
    vl.set_axis_from_slice(1, &[2.0, 3.0]);
    vl.fill(0.5);
    let norms = vl.get_norms();
    assert_eq!(norms.len(), 2);
    vl.normalize();
    let (_r, _u) = vl.to_polar();

    let _ = &vl + &vl;
    let _ = &vl - &vl;
    let _ = &vl * &vl;
    let _ = &vl / &vl;
    let mut vl2 = vl.clone();
    vl2 += &vl;
    vl2 -= &vl;
    vl2 *= &vl;
    vl2 /= &vl;
    let _ = &vl + 1.0;
    let _ = &vl - 1.0;
    let _ = &vl * 2.0;
    let _ = &vl / 2.0;
    vl2 += 1.0;
    vl2 -= 1.0;
    vl2 *= 2.0;
    vl2 /= 2.0;
}

#[test]
/// Annotation:
/// - Purpose: Executes `vector_list_rand_public_surface` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn vector_list_rand_public_surface() {
    let mut hv = HaarVectors::new(3, 4, Some(2));
    hv.refresh();
    let h_arr = hv.to_ndarray();
    let _hv2 = HaarVectors::from_ndarray(&h_arr);

    let mut nn = NNVectors::new(3, 4, Some(2));
    nn.refresh();
    let n_arr = nn.to_ndarray();
    let _nn2 = NNVectors::from_ndarray(&n_arr);

    // generic ndarray conversion trait path
    let _hv3 = <HaarVectors as NdarrayConvert>::from_ndarray(&h_arr);
    let _nn3 = <NNVectors as NdarrayConvert>::from_ndarray(&n_arr);
}

#[test]
#[should_panic]
/// Annotation:
/// - Purpose: Executes `matrix_trait_col_view_panics_on_dense_row_major` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn matrix_trait_col_view_panics_on_dense_row_major() {
    let m = Matrix::<f64>::empty(2, 2);
    let _ = MatrixTrait::col_view(&m, 0);
}

#[test]
#[should_panic]
/// Annotation:
/// - Purpose: Executes `matrix_trait_col_view_mut_panics_on_dense_row_major` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn matrix_trait_col_view_mut_panics_on_dense_row_major() {
    let mut m = Matrix::<f64>::empty(2, 2);
    let _ = MatrixTrait::col_view_mut(&mut m, 0);
}

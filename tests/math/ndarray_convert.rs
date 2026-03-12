use ndarray::{arr2, ArrayD, IxDyn};

use physics_in_parallel::math::{
    ndarray_convert::NdarrayConvert,
    tensor::{
        core::dense::Tensor as DenseTensor,
        core::sparse::Tensor as SparseTensor,
        rank_2::{
            matrix::dense::Matrix,
            vector_list::VectorList,
            vector_list_rand::{HaarVectors, NNVectors},
        },
    },
};

#[test]
/// Annotation:
/// - Purpose: Executes `dense_tensor_ndarray_roundtrip` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn dense_tensor_ndarray_roundtrip() {
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t = DenseTensor::<f64>::from_ndarray(&a);
    let b = t.to_ndarray();
    assert_eq!(a, b);

    // Trait path
    let t2 = <DenseTensor<f64> as NdarrayConvert>::from_ndarray(&a);
    assert_eq!(t2.to_ndarray(), a);
}

#[test]
/// Annotation:
/// - Purpose: Executes `sparse_tensor_ndarray_roundtrip` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn sparse_tensor_ndarray_roundtrip() {
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0]).unwrap();
    let s = SparseTensor::<f64>::from_ndarray(&a);
    let b = s.to_ndarray();
    assert_eq!(a, b);
}

#[test]
/// Annotation:
/// - Purpose: Executes `dense_matrix_ndarray_roundtrip` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn dense_matrix_ndarray_roundtrip() {
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

    let m = Matrix::<f64>::from_ndarray(&a);
    let b = m.to_ndarray();
    assert_eq!(a, b);
}

#[test]
/// Annotation:
/// - Purpose: Executes `vector_list_ndarray_roundtrip` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn vector_list_ndarray_roundtrip() {
    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]); // shape [n=2, dim=3]

    let vl = VectorList::<f64>::from_ndarray(&a);
    assert_eq!(vl.shape(), [2, 3]);
    let b = vl.to_ndarray();
    assert_eq!(a, b);
}

#[test]
/// Annotation:
/// - Purpose: Executes `vector_list_rand_ndarray_roundtrip` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn vector_list_rand_ndarray_roundtrip() {
    let hf_arr = arr2(&[[0.5, 0.5], [0.5, -0.5]]);
    let hf = HaarVectors::from_ndarray(&hf_arr);
    assert_eq!(hf.to_ndarray(), hf_arr);

    let nn_arr = arr2(&[[1isize, 0isize], [0isize, -1isize]]);
    let nn = NNVectors::from_ndarray(&nn_arr);
    assert_eq!(nn.to_ndarray(), nn_arr);
}

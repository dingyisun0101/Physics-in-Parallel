use ndarray::{Array1, Array2, array};

use physics_in_parallel::prelude::advanced::Dense;
use physics_in_parallel::prelude::basic::{Complex, Tensor};

fn dense_from_array2(array: &Array2<f64>) -> Tensor<f64, Dense> {
    let shape = array.shape();
    let mut tensor = Tensor::<f64, Dense>::empty(shape);
    for ((i, j), value) in array.indexed_iter() {
        tensor.set(&[i as isize, j as isize], *value);
    }
    tensor
}

fn dense_from_array1(array: &Array1<f64>) -> Tensor<f64, Dense> {
    let mut tensor = Tensor::<f64, Dense>::empty(&[array.len()]);
    for (i, value) in array.indexed_iter() {
        tensor.set(&[i as isize], *value);
    }
    tensor
}

fn assert_tensor_eq_array2(tensor: &Tensor<f64, Dense>, expected: &Array2<f64>) {
    assert_eq!(tensor.shape(), expected.shape());
    for ((i, j), value) in expected.indexed_iter() {
        assert_eq!(tensor.get(&[i as isize, j as isize]), *value);
    }
}

fn assert_tensor_eq_array1(tensor: &Tensor<f64, Dense>, expected: &Array1<f64>) {
    assert_eq!(tensor.shape(), expected.shape());
    for (i, value) in expected.indexed_iter() {
        assert_eq!(tensor.get(&[i as isize]), *value);
    }
}

#[test]
fn transpose_and_matmul_match_ndarray() {
    let a_ref = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b_ref = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
    let a = dense_from_array2(&a_ref);
    let b = dense_from_array2(&b_ref);

    let transpose = a.transpose();
    assert_tensor_eq_array2(&transpose, &a_ref.t().to_owned());

    let product = a.matmul(&b);
    assert_tensor_eq_array2(&product, &a_ref.dot(&b_ref));
}

#[test]
fn dot_cross_wedge_and_norm_match_ndarray_reference_calculations() {
    let a_ref = array![1.0, 2.0, 3.0];
    let b_ref = array![4.0, 5.0, 6.0];
    let a = dense_from_array1(&a_ref);
    let b = dense_from_array1(&b_ref);

    assert_eq!(a.dot(&b), a_ref.dot(&b_ref));
    assert_eq!(a.hermitian_dot(&b), a_ref.dot(&b_ref));
    assert_eq!(a.norm_sqr_real(), a_ref.dot(&a_ref));
    assert_eq!(a.norm(), a_ref.dot(&a_ref).sqrt());

    let cross_ref = array![
        a_ref[1] * b_ref[2] - a_ref[2] * b_ref[1],
        a_ref[2] * b_ref[0] - a_ref[0] * b_ref[2],
        a_ref[0] * b_ref[1] - a_ref[1] * b_ref[0],
    ];
    assert_tensor_eq_array1(&a.cross(&b), &cross_ref);

    let wedge_ref =
        Array2::from_shape_fn((3, 3), |(i, j)| a_ref[i] * b_ref[j] - a_ref[j] * b_ref[i]);
    assert_tensor_eq_array2(&a.wedge(&b), &wedge_ref);
}

#[test]
fn hermitian_transpose_and_dot_match_ndarray_style_reference() {
    let mut a = Tensor::<Complex<f64>, Dense>::empty(&[2, 2]);
    let a_ref = [
        [Complex::new(1.0, 1.0), Complex::new(2.0, -3.0)],
        [Complex::new(4.0, 5.0), Complex::new(6.0, -7.0)],
    ];
    for i in 0..2 {
        for j in 0..2 {
            a.set(&[i, j], a_ref[i as usize][j as usize]);
        }
    }

    let ah = a.hermitian_transpose();
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(ah.get(&[i, j]), a_ref[j as usize][i as usize].conj());
        }
    }

    let mut x = Tensor::<Complex<f64>, Dense>::empty(&[2]);
    let x_ref = [Complex::new(1.0, 1.0), Complex::new(2.0, -1.0)];
    x.set(&[0], x_ref[0]);
    x.set(&[1], x_ref[1]);

    let mut y = Tensor::<Complex<f64>, Dense>::empty(&[2]);
    let y_ref = [Complex::new(3.0, 0.0), Complex::new(0.0, 4.0)];
    y.set(&[0], y_ref[0]);
    y.set(&[1], y_ref[1]);

    let dot_ref = x_ref[0] * y_ref[0] + x_ref[1] * y_ref[1];
    let hermitian_dot_ref = x_ref[0].conj() * y_ref[0] + x_ref[1].conj() * y_ref[1];
    let norm_sqr_ref = x_ref[0].norm_sqr() + x_ref[1].norm_sqr();

    assert_eq!(x.dot(&y), dot_ref);
    assert_eq!(x.hermitian_dot(&y), hermitian_dot_ref);
    assert_eq!(x.norm_sqr_real(), norm_sqr_ref);
    assert_eq!(x.norm(), Complex::new(norm_sqr_ref.sqrt(), 0.0));
}

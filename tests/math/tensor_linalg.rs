use physics_in_parallel::prelude::advanced::Dense;
use physics_in_parallel::prelude::basic::{Complex, Tensor};

fn assert_tensor(tensor: &Tensor<f64, Dense>, shape: &[usize], expected: &[f64]) {
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.as_slice(), expected);
}

#[test]
fn transpose_and_matmul_match_manual_results() {
    let a = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::from_vec(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let transpose = a.transpose();
    assert_tensor(&transpose, &[3, 2], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let product = a.matmul(&b);
    assert_tensor(&product, &[2, 2], &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn dot_cross_wedge_and_norm_match_manual_calculations() {
    let a = Tensor::from_vec(&[3], vec![1.0, 2.0, 3.0]);
    let b = Tensor::from_vec(&[3], vec![4.0, 5.0, 6.0]);

    assert_eq!(a.dot(&b), 32.0);
    assert_eq!(a.hermitian_dot(&b), 32.0);
    assert_eq!(a.norm_sqr_real(), 14.0);
    assert_eq!(a.norm(), 14.0_f64.sqrt());

    assert_tensor(&a.cross(&b), &[3], &[-3.0, 6.0, -3.0]);

    assert_tensor(
        &a.wedge(&b),
        &[3, 3],
        &[0.0, -3.0, -6.0, 3.0, 0.0, -3.0, 6.0, 3.0, 0.0],
    );
}

#[test]
fn hermitian_transpose_and_dot_match_manual_reference() {
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

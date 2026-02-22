use num_complex::Complex;

use physics_in_parallel::math::{
    scalar::Scalar,
    tensor::core::{dense::Tensor as DenseTensor, tensor_trait::TensorTrait},
};

#[test]
/// Annotation:
/// - Purpose: Executes `signed_abs_and_sqrt_edge_cases` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn signed_abs_and_sqrt_edge_cases() {
    assert_eq!(i8::MIN.abs_real(), i8::MAX);
    assert_eq!(i16::MIN.abs_real(), i16::MAX);
    assert_eq!((-123i64).sqrt(), 0);
    assert_eq!(0i64.sqrt(), 0);
    assert_eq!(15i64.sqrt(), 3);
    assert_eq!(16i64.sqrt(), 4);
    assert_eq!(17i64.sqrt(), 4);
}

#[test]
/// Annotation:
/// - Purpose: Executes `unsigned_sqrt_large_is_exact_floor` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn unsigned_sqrt_large_is_exact_floor() {
    let n = u128::MAX;
    let r = n.sqrt();
    assert!(r.checked_mul(r).map_or(false, |x| x <= n));
    let upper_ok = match r.checked_add(1).and_then(|x| x.checked_mul(x)) {
        Some(x) => x > n,
        None => true,
    };
    assert!(upper_ok);
}

#[test]
/// Annotation:
/// - Purpose: Executes `complex_scalar_ops_basic` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn complex_scalar_ops_basic() {
    let z = Complex::<f64>::from_re_im(3.0, 4.0);
    assert_eq!(z.re(), 3.0);
    assert_eq!(z.im(), 4.0);
    assert_eq!(z.conj(), Complex::new(3.0, -4.0));
    assert!((z.abs_real() - 5.0).abs() < 1e-12);
}

#[test]
/// Annotation:
/// - Purpose: Executes `dense_cast_complex_to_real_drops_imag_part` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn dense_cast_complex_to_real_drops_imag_part() {
    let t = DenseTensor::<Complex<f64>> {
        shape: vec![2],
        data: vec![Complex::new(1.25, 9.0), Complex::new(-2.5, -3.0)],
    };

    let r = t.cast_to::<f64>();
    assert_eq!(r.shape, vec![2]);
    assert_eq!(r.data, vec![1.25, -2.5]);
}

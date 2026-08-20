use physics_in_parallel::prelude::basic::{Complex, Scalar, ScalarCastError};

fn assert_close_f64(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}"
    );
}

fn assert_close_complex(actual: Complex<f64>, expected: Complex<f64>) {
    assert_close_f64(actual.re, expected.re);
    assert_close_f64(actual.im, expected.im);
}

#[test]
fn real_float_scalar_algebra_is_type_preserving() {
    let x = -3.5f64;
    let y = 2.0f64;

    assert_eq!(x + y, -1.5);
    assert_eq!(x - y, -5.5);
    assert_eq!(x * y, -7.0);
    assert_eq!(x / y, -1.75);
    assert_eq!(-x, 3.5);

    let conj: f64 = x.conj();
    let abs: f64 = x.abs();
    let norm_sqr: f64 = x.norm_sqr();
    let sqrt: f64 = 9.0f64.sqrt();

    assert_eq!(conj, x);
    assert_eq!(abs, 3.5);
    assert_eq!(norm_sqr, 12.25);
    assert_eq!(sqrt, 3.0);
}

#[test]
fn real_float_scalar_projection_and_construction() {
    let x = -7.25f64;

    assert_eq!(x.re(), -7.25);
    assert_eq!(x.im(), 0.0);
    assert_eq!(x.abs_real(), 7.25);
    assert_eq!(x.norm_sqr_real(), 52.5625);
    assert_eq!(f64::from_re_im(3.0, 99.0), 3.0);

    assert!(x.is_finite());
    assert!(!f64::INFINITY.is_finite());
    assert!(!f64::NAN.is_finite());
}

#[test]
fn signed_integer_scalar_algebra_uses_total_integer_semantics() {
    let x = -12i64;
    let y = 5i64;

    assert_eq!(x + y, -7);
    assert_eq!(x - y, -17);
    assert_eq!(x * y, -60);
    assert_eq!(x / -3, 4);
    assert_eq!(-y, -5);

    let conj: i64 = x.conj();
    let abs: i64 = x.abs();
    let norm_sqr: i64 = x.norm_sqr();
    let sqrt: i64 = 80i64.sqrt();

    assert_eq!(conj, x);
    assert_eq!(abs, 12);
    assert_eq!(norm_sqr, 144);
    assert_eq!(sqrt, 8);
    assert_eq!((-1i64).sqrt(), 0);
}

#[test]
fn signed_integer_projection_and_saturation_edges() {
    assert_eq!((-12i64).re(), -12);
    assert_eq!((-12i64).im(), 0);
    assert_eq!((-12i64).abs_real(), 12);
    assert_eq!((-12i64).norm_sqr_real(), 144);
    assert_eq!(i64::from_re_im(7, -99), 7);
    assert!(i64::MAX.is_finite());

    assert_eq!(<i8 as Scalar>::abs(i8::MIN), i8::MAX);
    assert_eq!(i8::MIN.abs_real(), i8::MAX);
    assert_eq!(<i8 as Scalar>::norm_sqr(i8::MIN), i8::MAX);
    assert_eq!(i8::MIN.norm_sqr_real(), i8::MAX);
}

#[test]
fn unsigned_integer_scalar_algebra_uses_floor_sqrt_and_saturation() {
    let x = 12u64;
    let y = 5u64;

    assert_eq!(x + y, 17);
    assert_eq!(x - y, 7);
    assert_eq!(x * y, 60);
    assert_eq!(x / 3, 4);

    assert_eq!(x.conj(), x);
    assert_eq!(x.abs(), x);
    assert_eq!(x.norm_sqr(), 144);
    assert_eq!(80u64.sqrt(), 8);
    assert_eq!(u8::MAX.norm_sqr(), u8::MAX);
    assert_eq!(u64::from_re_im(9, 99), 9);
    assert!(u64::MAX.is_finite());
}

#[test]
fn complex_scalar_algebra_is_type_preserving() {
    let z = Complex::<f64>::new(3.0, 4.0);
    let w = Complex::<f64>::new(1.0, -2.0);

    assert_eq!(z + w, Complex::new(4.0, 2.0));
    assert_eq!(z - w, Complex::new(2.0, 6.0));
    assert_eq!(z * w, Complex::new(11.0, -2.0));
    assert_close_complex(z / w, Complex::new(-1.0, 2.0));
    assert_eq!(-z, Complex::new(-3.0, -4.0));

    let conj: Complex<f64> = z.conj();
    let abs: Complex<f64> = z.abs();
    let norm_sqr: Complex<f64> = z.norm_sqr();
    let sqrt: Complex<f64> = z.sqrt();

    assert_eq!(conj, Complex::new(3.0, -4.0));
    assert_eq!(abs, Complex::new(5.0, 0.0));
    assert_eq!(norm_sqr, Complex::new(25.0, 0.0));
    assert_close_complex(sqrt * sqrt, z);
}

#[test]
fn complex_projection_construction_and_finiteness() {
    let z = Complex::<f64>::from_re_im(3.0, 4.0);

    assert_eq!(z.re(), 3.0);
    assert_eq!(z.im(), 4.0);
    assert_eq!(z.abs_real(), 5.0);
    assert_eq!(z.norm_sqr_real(), 25.0);
    assert!(z.is_finite());
    assert!(!Complex::<f64>::new(f64::INFINITY, 0.0).is_finite());
    assert!(!Complex::<f64>::new(0.0, f64::NAN).is_finite());
}

#[test]
fn scalar_casts_cover_real_integer_and_complex_boundaries() {
    let float: f64 = 42i64.cast::<f64>();
    let int: i64 = 12.0f64.try_cast::<i64>().unwrap();
    let complex: Complex<f64> = 7i64.cast::<Complex<f64>>();
    let complex32: Complex<f32> = Complex::<f64>::new(1.25, -2.5)
        .try_cast::<Complex<f32>>()
        .unwrap();
    let real: f64 = Complex::<f64>::new(9.5, -3.0).cast::<f64>();

    assert_eq!(float, 42.0);
    assert_eq!(int, 12);
    assert_eq!(complex, Complex::new(7.0, 0.0));
    assert_eq!(complex32, Complex::<f32>::new(1.25, -2.5));
    assert_eq!(real, 9.5);

    let real_err = i64::MAX.try_cast::<i8>().unwrap_err();
    assert!(matches!(
        real_err,
        ScalarCastError::RealPartOutOfRange { .. }
    ));

    let imag_err = Complex::<f64>::new(0.0, f64::MAX)
        .try_cast::<Complex<f32>>()
        .unwrap_err();
    assert!(matches!(
        imag_err,
        ScalarCastError::ImagPartOutOfRange { .. }
    ));
}

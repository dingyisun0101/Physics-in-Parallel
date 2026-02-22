//! Scalar trait implementations for complex types.

use num_complex::Complex;

use super::scalar_trait::{
    scalar_sealed::Sealed,
    Scalar,
};

impl Sealed for Complex<f32> {}
impl Sealed for Complex<f64> {}

impl Scalar for Complex<f32> {
    type Real = f32;

    #[inline] fn conj(self) -> Self { Complex::conj(&self) }
    #[inline] fn re(self) -> Self::Real { self.re }
    #[inline] fn im(self) -> Self::Real { self.im }
    #[inline] fn from_re_im(re: Self::Real, im: Self::Real) -> Self { Complex::new(re, im) }

    #[inline] fn abs_real(self) -> Self::Real { self.norm() }
    #[inline] fn norm_sqr_real(self) -> Self::Real { self.norm_sqr() }
    #[inline] fn sqrt(self) -> Self { Complex::sqrt(self) }
    #[inline] fn is_finite(self) -> bool { self.re.is_finite() && self.im.is_finite() }
}

impl Scalar for Complex<f64> {
    type Real = f64;

    #[inline] fn conj(self) -> Self { Complex::conj(&self) }
    #[inline] fn re(self) -> Self::Real { self.re }
    #[inline] fn im(self) -> Self::Real { self.im }
    #[inline] fn from_re_im(re: Self::Real, im: Self::Real) -> Self { Complex::new(re, im) }

    #[inline] fn abs_real(self) -> Self::Real { self.norm() }
    #[inline] fn norm_sqr_real(self) -> Self::Real { self.norm_sqr() }
    #[inline] fn sqrt(self) -> Self { Complex::sqrt(self) }
    #[inline] fn is_finite(self) -> bool { self.re.is_finite() && self.im.is_finite() }
}


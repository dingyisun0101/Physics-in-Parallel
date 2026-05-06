//! Scalar implementations for complex primitive numeric types.
//!
//! Complex values use the associated real type as their projection domain:
//!	- `Complex<f32>::Real = f32`;
//!	- `Complex<f64>::Real = f64`.
//!
//! Type-preserving helpers inherited from `Scalar`, such as `abs()` and
//! `norm_sqr()`, return complex values with the result in the real component
//! and zero in the imaginary component. Use `abs_real()` and `norm_sqr_real()`
//! when the real-valued mathematical result is desired directly.

use num_complex::Complex;

use super::scalar_trait::{Scalar, scalar_sealed::Sealed};

impl Sealed for Complex<f32> {}
impl Sealed for Complex<f64> {}

impl Scalar for Complex<f32> {
    type Real = f32;

    #[inline]
    fn conj(self) -> Self {
        Complex::conj(&self)
    }

    #[inline]
    fn re(self) -> Self::Real {
        self.re
    }

    #[inline]
    fn im(self) -> Self::Real {
        self.im
    }

    #[inline]
    fn from_re_im(re: Self::Real, im: Self::Real) -> Self {
        Complex::new(re, im)
    }

    #[inline]
    fn abs_real(self) -> Self::Real {
        self.norm()
    }

    #[inline]
    fn norm_sqr_real(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn sqrt(self) -> Self {
        Complex::sqrt(self)
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl Scalar for Complex<f64> {
    type Real = f64;

    #[inline]
    fn conj(self) -> Self {
        Complex::conj(&self)
    }

    #[inline]
    fn re(self) -> Self::Real {
        self.re
    }

    #[inline]
    fn im(self) -> Self::Real {
        self.im
    }

    #[inline]
    fn from_re_im(re: Self::Real, im: Self::Real) -> Self {
        Complex::new(re, im)
    }

    #[inline]
    fn abs_real(self) -> Self::Real {
        self.norm()
    }

    #[inline]
    fn norm_sqr_real(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn sqrt(self) -> Self {
        Complex::sqrt(self)
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

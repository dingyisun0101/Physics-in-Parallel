// src/field/scalar_complex.rs
use core::fmt::{Debug, Display};
use core::iter::{Product, Sum};
use num_complex::Complex;
use num_traits::{Num, NumCast, One, Zero};

use super::scalar_real::ScalarReal;

// —————————————————————————————————————————————————————————————
// Sealed to keep the impl surface controlled
// —————————————————————————————————————————————————————————————
mod sealed {
    pub trait Sealed {}
    impl Sealed for num_complex::Complex<f32> {}
    impl Sealed for num_complex::Complex<f64> {}
}
use sealed::Sealed;

/// The single public trait for complex scalars.
///
/// Goals:
/// - Parallel-safe (`Send + Sync + 'static`)
/// - Type associates its real part via `Real`
/// - Uniform math API: conj, |z|, |z|^2, arg, polar, sqrt (principal)
pub trait ScalarComplex:
    Num
    + NumCast
    + Zero
    + One
    + Copy
    + Send
    + Sync
    + 'static
    + Debug
    + Display
    + Sum<Self>
    + Product<Self>
    + Sealed
{
    /// The corresponding real scalar type (`f32` or `f64`).
    type Real: ScalarReal;

    /// Complex conjugate.
    fn conj(self) -> Self;

    /// Magnitude (Euclidean norm) as a real.
    fn abs_real(self) -> Self::Real;

    /// Squared magnitude as a real (no sqrt).
    fn norm_sqr(self) -> Self::Real;

    /// Argument (phase) in radians, principal value in (-π, π].
    fn arg(self) -> Self::Real;

    /// Return `(r, theta)` where `r = |z|`, `theta = arg(z)`.
    #[inline]
    fn to_polar(self) -> (Self::Real, Self::Real) {
        (self.abs_real(), self.arg())
    }

    /// Principal complex square root.
    fn sqrt_complex(self) -> Self;

    /// Real/imag accessors.
    fn re(self) -> Self::Real;
    fn im(self) -> Self::Real;

    /// Construct from real & imaginary parts.
    fn from_re_im(re: Self::Real, im: Self::Real) -> Self;

    /// Finite check (all parts finite).
    fn is_finite_c(self) -> bool;
}

// ———————————————— Impl for Complex<f32> ————————————————
impl ScalarComplex for Complex<f32> {
    type Real = f32;

    #[inline] fn conj(self) -> Self { self.conj() }
    #[inline] fn abs_real(self) -> Self::Real { self.norm() }
    #[inline] fn norm_sqr(self) -> Self::Real { self.norm_sqr() }
    #[inline] fn arg(self) -> Self::Real { self.arg() }
    #[inline] fn sqrt_complex(self) -> Self { self.sqrt() }
    #[inline] fn re(self) -> Self::Real { self.re }
    #[inline] fn im(self) -> Self::Real { self.im }
    #[inline] fn from_re_im(re: Self::Real, im: Self::Real) -> Self { Complex::new(re, im) }
    #[inline] fn is_finite_c(self) -> bool { self.re.is_finite() && self.im.is_finite() }
}

// ———————————————— Impl for Complex<f64> ————————————————
impl ScalarComplex for Complex<f64> {
    type Real = f64;

    #[inline] fn conj(self) -> Self { self.conj() }
    #[inline] fn abs_real(self) -> Self::Real { self.norm() }
    #[inline] fn norm_sqr(self) -> Self::Real { self.norm_sqr() }
    #[inline] fn arg(self) -> Self::Real { self.arg() }
    #[inline] fn sqrt_complex(self) -> Self { self.sqrt() }
    #[inline] fn re(self) -> Self::Real { self.re }
    #[inline] fn im(self) -> Self::Real { self.im }
    #[inline] fn from_re_im(re: Self::Real, im: Self::Real) -> Self { Complex::new(re, im) }
    #[inline] fn is_finite_c(self) -> bool { self.re.is_finite() && self.im.is_finite() }
}


//! A single minimal `Scalar` trait that unifies real (ints/floats) and complex scalars.

use core::fmt::{Debug, Display};
use core::iter::{Product, Sum};
use num_complex::Complex;
use num_traits::{Num, NumCast, One, Zero};
use sealed::Sealed;
use serde::Serialize;
use serde::de::DeserializeOwned;

// ==============================================================================
// ------------------- Sealing: keep impl surface controlled --------------------
// ==============================================================================
mod sealed {
    pub trait Sealed {}
    macro_rules! impl_sealed_for {
        ($($t:ty),* $(,)?) => { $(impl Sealed for $t {})* };
    }
    impl_sealed_for!(
        // unsigned
        u8, u16, u32, u64, u128, usize,
        // signed
        i8, i16, i32, i64, i128, isize,
        // floats
        f32, f64,
        // complex
        num_complex::Complex<f32>, num_complex::Complex<f64>
    );
}

// ==============================================================================
// --------------------------------- Trait Def ----------------------------------
// ==============================================================================
/**
A minimal, unified scalar trait for reals (ints/floats) and complex numbers.

Notes:
- No lifetime parameter. We use `DeserializeOwned` so implementors can be deserialized.
- `Real = Self` for reals; `Real = f32/f64` for complex.
*/
pub trait Scalar:
    Num
    + NumCast
    + Zero
    + One
    + Copy
    + Clone
    + Default
    + Send
    + Sync
    + 'static
    + Debug
    + Display
    + Sum<Self>
    + Product<Self>
    + Sealed
    + Serialize
    + DeserializeOwned
{
    /// The associated *real* type:
    /// - reals: `Real = Self`
    /// - complex: `Real = f32` or `f64`
    type Real: Num
        + NumCast
        + Zero
        + One
        + Copy
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + Debug
        + Display
        + Sum<Self::Real>
        + Product<Self::Real>;

    /// Complex conjugate (identity for reals).
    fn conj(self) -> Self;

    /// Real part (`self` for reals).
    fn re(self) -> Self::Real;

    /// Imag part (0 for reals).
    fn im(self) -> Self::Real;

    /// Construct from real/imag parts (imag ignored for reals).
    fn from_re_im(re: Self::Real, im: Self::Real) -> Self;

    /// Magnitude (Euclidean norm) as a real: `|x|` for reals, `|z|` for complex.
    fn abs_real(self) -> Self::Real;

    /// Squared magnitude as a real: `x*x` for reals, `|z|^2` for complex.
    fn norm_sqr_real(self) -> Self::Real;

    /// Type-preserving square root (see header doc).
    fn sqrt(self) -> Self;

    /// Finite check (floats/complex parts use native finite; integers are always finite).
    fn is_finite(self) -> bool;
}

// ==============================================================================
// -------------------------------- IMPL: Real ----------------------------------
// ==============================================================================

// ----------- Float ----------------
impl Scalar for f32 {
    type Real = f32;

    #[inline] fn conj(self) -> Self { self }
    #[inline] fn re(self) -> Self::Real { self }
    #[inline] fn im(self) -> Self::Real { 0.0 }
    #[inline] fn from_re_im(re: Self::Real, _im: Self::Real) -> Self { re }

    #[inline] fn abs_real(self) -> Self::Real { self.abs() }
    #[inline] fn norm_sqr_real(self) -> Self::Real { self * self }
    #[inline] fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline] fn is_finite(self) -> bool { f32::is_finite(self) }
}

impl Scalar for f64 {
    type Real = f64;

    #[inline] fn conj(self) -> Self { self }
    #[inline] fn re(self) -> Self::Real { self }
    #[inline] fn im(self) -> Self::Real { 0.0 }
    #[inline] fn from_re_im(re: Self::Real, _im: Self::Real) -> Self { re }

    #[inline] fn abs_real(self) -> Self::Real { self.abs() }
    #[inline] fn norm_sqr_real(self) -> Self::Real { self * self }
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline] fn is_finite(self) -> bool { f64::is_finite(self) }
}

// ----------- Unsigned ----------------
macro_rules! impl_scalar_unsigned {
    ($($t:ty),* $(,)?) => {$(
        impl Scalar for $t {
            type Real = $t;

            #[inline] fn conj(self) -> Self { self }
            #[inline] fn re(self) -> Self::Real { self }
            #[inline] fn im(self) -> Self::Real { 0 as $t }
            #[inline] fn from_re_im(re: Self::Real, _im: Self::Real) -> Self { re }

            #[inline] fn abs_real(self) -> Self::Real { self }
            #[inline] fn norm_sqr_real(self) -> Self::Real { self.saturating_mul(self) }
            #[inline] fn sqrt(self) -> Self {
                // floor(sqrt(x)) via fast f64 path
                (self as f64).sqrt().floor() as $t
            }
            #[inline] fn is_finite(self) -> bool { true }
        }
    )*}
}
impl_scalar_unsigned!(u8, u16, u32, u64, u128, usize);

// ----------- Signed ----------------
macro_rules! impl_scalar_signed {
    ($($t:ty),* $(,)?) => {$(
        impl Scalar for $t {
            type Real = $t;

            #[inline] fn conj(self) -> Self { self }
            #[inline] fn re(self) -> Self::Real { self }
            #[inline] fn im(self) -> Self::Real { 0 as $t }
            #[inline] fn from_re_im(re: Self::Real, _im: Self::Real) -> Self { re }

            #[inline]
            fn abs_real(self) -> Self::Real {
                if self >= 0 { self } else { (-(self as f64)) as $t }
            }

            #[inline]
            fn norm_sqr_real(self) -> Self::Real {
                self.saturating_mul(self)
            }

            #[inline]
            fn sqrt(self) -> Self {
                if self <= 0 { 0 as $t }
                else { (self as f64).sqrt().floor() as $t }
            }

            #[inline] fn is_finite(self) -> bool { true }
        }
    )*}
}
impl_scalar_signed!(i8, i16, i32, i64, i128, isize);

// ==============================================================================
// ------------------------------- IMPL: Complex --------------------------------
// ==============================================================================
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

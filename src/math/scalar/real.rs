//! Scalar trait implementations for real primitive types.

use super::scalar_trait::{
    scalar_sealed::Sealed,
    Scalar,
};

#[inline]
fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut lo = 0u128;
    let mut hi = 1u128 << 64; // sqrt(u128::MAX) < 2^64
    while lo + 1 < hi {
        let mid = lo + ((hi - lo) >> 1);
        if mid <= n / mid {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

macro_rules! impl_sealed_for {
    ($($t:ty),* $(,)?) => { $(impl Sealed for $t {})* };
}

impl_sealed_for!(
    // unsigned
    u8, u16, u32, u64, u128, usize,
    // signed
    i8, i16, i32, i64, i128, isize,
    // floats
    f32, f64
);

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
            #[inline] fn sqrt(self) -> Self { isqrt_u128(self as u128) as $t }
            #[inline] fn is_finite(self) -> bool { true }
        }
    )*}
}

impl_scalar_unsigned!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_scalar_signed {
    ($($t:ty),* $(,)?) => {$(
        impl Scalar for $t {
            type Real = $t;

            #[inline] fn conj(self) -> Self { self }
            #[inline] fn re(self) -> Self::Real { self }
            #[inline] fn im(self) -> Self::Real { 0 as $t }
            #[inline] fn from_re_im(re: Self::Real, _im: Self::Real) -> Self { re }

            #[inline] fn abs_real(self) -> Self::Real { self.saturating_abs() }
            #[inline] fn norm_sqr_real(self) -> Self::Real { self.saturating_mul(self) }
            #[inline]
            fn sqrt(self) -> Self {
                if self <= 0 {
                    0 as $t
                } else {
                    isqrt_u128(self as u128) as $t
                }
            }
            #[inline] fn is_finite(self) -> bool { true }
        }
    )*}
}

impl_scalar_signed!(i8, i16, i32, i64, i128, isize);


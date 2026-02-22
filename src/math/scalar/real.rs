//! Scalar trait implementations for real primitive types.

use super::scalar_trait::{
    scalar_sealed::Sealed,
    Scalar,
};

#[inline]
/// Annotation:
/// - Purpose: Executes `isqrt_u128` logic for this module.
/// - Parameters:
///   - `n` (`u128`): Parameter of type `u128` used by `isqrt_u128`.
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

    /// Annotation:
    /// - Purpose: Executes `conj` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn conj(self) -> Self { self }
    /// Annotation:
    /// - Purpose: Executes `re` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn re(self) -> Self::Real { self }
    /// Annotation:
    /// - Purpose: Executes `im` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn im(self) -> Self::Real { 0.0 }
    /// Annotation:
    /// - Purpose: Builds this value from `re_im` input.
    /// - Parameters:
    ///   - `re` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
    ///   - `_im` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
    #[inline] fn from_re_im(re: Self::Real, _im: Self::Real) -> Self { re }

    /// Annotation:
    /// - Purpose: Executes `abs_real` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn abs_real(self) -> Self::Real { self.abs() }
    /// Annotation:
    /// - Purpose: Executes `norm_sqr_real` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn norm_sqr_real(self) -> Self::Real { self * self }
    /// Annotation:
    /// - Purpose: Executes `sqrt` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn sqrt(self) -> Self { f32::sqrt(self) }
    /// Annotation:
    /// - Purpose: Checks whether `finite` condition is true.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn is_finite(self) -> bool { f32::is_finite(self) }
}

impl Scalar for f64 {
    type Real = f64;

    /// Annotation:
    /// - Purpose: Executes `conj` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn conj(self) -> Self { self }
    /// Annotation:
    /// - Purpose: Executes `re` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn re(self) -> Self::Real { self }
    /// Annotation:
    /// - Purpose: Executes `im` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn im(self) -> Self::Real { 0.0 }
    /// Annotation:
    /// - Purpose: Builds this value from `re_im` input.
    /// - Parameters:
    ///   - `re` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
    ///   - `_im` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
    #[inline] fn from_re_im(re: Self::Real, _im: Self::Real) -> Self { re }

    /// Annotation:
    /// - Purpose: Executes `abs_real` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn abs_real(self) -> Self::Real { self.abs() }
    /// Annotation:
    /// - Purpose: Executes `norm_sqr_real` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn norm_sqr_real(self) -> Self::Real { self * self }
    /// Annotation:
    /// - Purpose: Executes `sqrt` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    /// Annotation:
    /// - Purpose: Checks whether `finite` condition is true.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn is_finite(self) -> bool { f64::is_finite(self) }
}

macro_rules! impl_scalar_unsigned {
    ($($t:ty),* $(,)?) => {$(
        impl Scalar for $t {
            type Real = $t;

            /// Annotation:
            /// - Purpose: Executes `conj` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn conj(self) -> Self { self }
            /// Annotation:
            /// - Purpose: Executes `re` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn re(self) -> Self::Real { self }
            /// Annotation:
            /// - Purpose: Executes `im` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn im(self) -> Self::Real { 0 as $t }
            /// Annotation:
            /// - Purpose: Builds this value from `re_im` input.
            /// - Parameters:
            ///   - `re` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
            ///   - `_im` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
            #[inline] fn from_re_im(re: Self::Real, _im: Self::Real) -> Self { re }

            /// Annotation:
            /// - Purpose: Executes `abs_real` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn abs_real(self) -> Self::Real { self }
            /// Annotation:
            /// - Purpose: Executes `norm_sqr_real` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn norm_sqr_real(self) -> Self::Real { self.saturating_mul(self) }
            /// Annotation:
            /// - Purpose: Executes `sqrt` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn sqrt(self) -> Self { isqrt_u128(self as u128) as $t }
            /// Annotation:
            /// - Purpose: Checks whether `finite` condition is true.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn is_finite(self) -> bool { true }
        }
    )*}
}

impl_scalar_unsigned!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_scalar_signed {
    ($($t:ty),* $(,)?) => {$(
        impl Scalar for $t {
            type Real = $t;

            /// Annotation:
            /// - Purpose: Executes `conj` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn conj(self) -> Self { self }
            /// Annotation:
            /// - Purpose: Executes `re` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn re(self) -> Self::Real { self }
            /// Annotation:
            /// - Purpose: Executes `im` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn im(self) -> Self::Real { 0 as $t }
            /// Annotation:
            /// - Purpose: Builds this value from `re_im` input.
            /// - Parameters:
            ///   - `re` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
            ///   - `_im` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
            #[inline] fn from_re_im(re: Self::Real, _im: Self::Real) -> Self { re }

            /// Annotation:
            /// - Purpose: Executes `abs_real` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn abs_real(self) -> Self::Real { self.saturating_abs() }
            /// Annotation:
            /// - Purpose: Executes `norm_sqr_real` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn norm_sqr_real(self) -> Self::Real { self.saturating_mul(self) }
            #[inline]
            /// Annotation:
            /// - Purpose: Executes `sqrt` logic for this module.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            fn sqrt(self) -> Self {
                if self <= 0 {
                    0 as $t
                } else {
                    isqrt_u128(self as u128) as $t
                }
            }
            /// Annotation:
            /// - Purpose: Checks whether `finite` condition is true.
            /// - Parameters:
            ///   - (none): This function has no documented non-receiver parameters.
            #[inline] fn is_finite(self) -> bool { true }
        }
    )*}
}

impl_scalar_signed!(i8, i16, i32, i64, i128, isize);


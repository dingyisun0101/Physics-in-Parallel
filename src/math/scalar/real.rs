//! Scalar implementations for real primitive numeric types.
//!
//! Integer semantics are intentionally total and type-preserving:
//!	- `sqrt` returns the floor integer square root for positive values;
//!	- signed negative `sqrt` returns zero;
//!	- signed `abs_real` uses saturating absolute value;
//!	- integer `norm_sqr_real` uses saturating multiplication.
//!
//! These rules keep generic scalar code usable for integer-backed tensors,
//! grids, labels, masks, and other discrete simulation data while preserving
//! the crate-wide `T -> T` scalar operation contract.

use super::scalar_trait::{Scalar, scalar_sealed::Sealed};

macro_rules! impl_sealed_for {
    ($($t:ty),* $(,)?) => { $(impl Sealed for $t {})* };
}

impl_sealed_for!(
    // unsigned
    u8, u16, u32, u64, u128, usize, // signed
    i8, i16, i32, i64, i128, isize
);

macro_rules! impl_float_kernels {
    ($ty:ty, $binary:ident, $scale:ident, $owned:ident) => {
        impl Sealed for $ty {
            fn scaled_values(input: &[Self], scalar: Self) -> Vec<Self> {
                crate::math::kernels::$owned(input, scalar)
            }
            fn binary_into(
                left: &[Self],
                right: &[Self],
                output: &mut [Self],
                op: crate::math::kernels::BinaryOp,
            ) {
                crate::math::kernels::$binary(left, right, output, op);
            }
            fn scale_slice(values: &mut [Self], scalar: Self) {
                crate::math::kernels::$scale(values, scalar);
            }
        }
    };
}
impl_float_kernels!(f32, binary_f32, scale_f32, scaled_f32);
impl_float_kernels!(f64, binary_f64, scale_f64, scaled_f64);

impl Scalar for f32 {
    type Real = f32;

    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn re(self) -> Self::Real {
        self
    }

    #[inline]
    fn im(self) -> Self::Real {
        0.0
    }

    #[inline]
    fn from_re_im(re: Self::Real, _im: Self::Real) -> Self {
        re
    }

    #[inline]
    fn abs_real(self) -> Self::Real {
        self.abs()
    }

    #[inline]
    fn norm_sqr_real(self) -> Self::Real {
        self * self
    }

    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

impl Scalar for f64 {
    type Real = f64;

    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn re(self) -> Self::Real {
        self
    }

    #[inline]
    fn im(self) -> Self::Real {
        0.0
    }

    #[inline]
    fn from_re_im(re: Self::Real, _im: Self::Real) -> Self {
        re
    }

    #[inline]
    fn abs_real(self) -> Self::Real {
        self.abs()
    }

    #[inline]
    fn norm_sqr_real(self) -> Self::Real {
        self * self
    }

    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

macro_rules! impl_scalar_unsigned {
    ($($t:ty),* $(,)?) => {$(
        impl Scalar for $t {
            type Real = $t;

            #[inline]
            fn conj(self) -> Self {
                self
            }

            #[inline]
            fn re(self) -> Self::Real {
                self
            }

            #[inline]
            fn im(self) -> Self::Real {
                0 as $t
            }

            #[inline]
            fn from_re_im(re: Self::Real, _im: Self::Real) -> Self {
                re
            }

            #[inline]
            fn abs_real(self) -> Self::Real {
                self
            }

            #[inline]
            fn norm_sqr_real(self) -> Self::Real {
                self.saturating_mul(self)
            }

            #[inline]
            fn sqrt(self) -> Self {
                self.isqrt()
            }

            #[inline]
            fn is_finite(self) -> bool {
                true
            }
        }
    )*};
}

impl_scalar_unsigned!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_scalar_signed {
    ($($t:ty),* $(,)?) => {$(
        impl Scalar for $t {
            type Real = $t;

            #[inline]
            fn conj(self) -> Self {
                self
            }

            #[inline]
            fn re(self) -> Self::Real {
                self
            }

            #[inline]
            fn im(self) -> Self::Real {
                0 as $t
            }

            #[inline]
            fn from_re_im(re: Self::Real, _im: Self::Real) -> Self {
                re
            }

            #[inline]
            fn abs_real(self) -> Self::Real {
                self.saturating_abs()
            }

            #[inline]
            fn norm_sqr_real(self) -> Self::Real {
                self.saturating_mul(self)
            }

            #[inline]
            fn sqrt(self) -> Self {
                if self <= 0 {
                    0 as $t
                } else {
                    self.isqrt()
                }
            }

            #[inline]
            fn is_finite(self) -> bool {
                true
            }
        }
    )*};
}

impl_scalar_signed!(i8, i16, i32, i64, i128, isize);

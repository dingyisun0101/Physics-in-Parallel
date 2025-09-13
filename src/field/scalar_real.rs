// src/field/scalar_real.rs
use core::fmt::{Debug, Display};
use core::iter::{Product, Sum};
use num_traits::{Bounded, Num, NumCast, One, Zero};

/// ================================================================================
/// ==============================  Seal: Impl Cleanup =============================
/// ================================================================================
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
        f32, f64
    );
}
use sealed::Sealed;


/// ================================================================================
/// ========================  Real Scalar: Trait Definition ========================
/// ================================================================================

/// The single public trait for “real scalars”.
///     - Works for *both* integers and floats
///     - Parallel-safe (`Send + Sync`)
///     - Provides a unified API, including a type-preserving sqrt
pub trait ScalarReal:
    Num
    + NumCast
    + Zero
    + One
    + Bounded
    + PartialOrd
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
    /// Type-preserving square root:
    /// - floats: exact `sqrt()`
    /// - unsigned ints: floor(sqrt(x))
    /// - signed ints: x <= 0 -> 0, else floor(sqrt(x))
    fn sqrt_real(self) -> Self;

    /// Optional convenience: `is_finite` for a uniform API.
    /// - floats use native `is_finite()`
    /// - integers are always finite
    #[inline]
    fn is_finite_real(self) -> bool { true }
}

// Blanket impl for all primitive types + specialized `sqrt_real`

// Floats
impl ScalarReal for f32 {
    #[inline] fn sqrt_real(self) -> Self { self.sqrt() }
    #[inline] fn is_finite_real(self) -> bool { self.is_finite() }
}
impl ScalarReal for f64 {
    #[inline] fn sqrt_real(self) -> Self { self.sqrt() }
    #[inline] fn is_finite_real(self) -> bool { self.is_finite() }
}

// Unsigned ints: floor(sqrt(x)) via fast f64 path
macro_rules! impl_scalar_unsigned {
    ($($t:ty),* $(,)?) => {$(
        impl ScalarReal for $t {
            #[inline]
            fn sqrt_real(self) -> Self {
                // floor(sqrt(x)) with a fast approximate path
                (self as f64).sqrt().floor() as $t
            }
        }
    )*}
}
impl_scalar_unsigned!(u8,u16,u32,u64,u128,usize);

// Signed ints: negatives -> 0, else floor(sqrt(x))
macro_rules! impl_scalar_signed {
    ($($t:ty),* $(,)?) => {$(
        impl ScalarReal for $t {
            #[inline]
            fn sqrt_real(self) -> Self {
                if self <= 0 { 0 as $t }
                else { (self as f64).sqrt().floor() as $t }
            }
        }
    )*}
}
impl_scalar_signed!(i8,i16,i32,i64,i128,isize);

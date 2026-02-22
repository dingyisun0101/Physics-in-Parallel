//! Core scalar traits shared by real and complex implementations.

use core::fmt::{Debug, Display};
use core::iter::{Product, Sum};

use num_traits::{Num, NumCast, One, Zero};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Internal sealing so only this crate can implement scalar traits.
pub(crate) mod scalar_sealed {
    pub trait Sealed {}
}

use scalar_sealed::Sealed;

/**
Compute-oriented scalar trait for real and complex numeric types.

`Real` is:
- `Self` for real scalars
- `f32`/`f64` for complex scalars
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
{
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
    /// Annotation:
    /// - Purpose: Executes `conj` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn conj(self) -> Self;

    /// Real part (`self` for reals).
    /// Annotation:
    /// - Purpose: Executes `re` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn re(self) -> Self::Real;

    /// Imaginary part (`0` for reals).
    /// Annotation:
    /// - Purpose: Executes `im` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn im(self) -> Self::Real;

    /// Construct from real/imaginary parts (imag ignored for reals).
    /// Annotation:
    /// - Purpose: Builds this value from `re_im` input.
    /// - Parameters:
    ///   - `re` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
    ///   - `im` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
    fn from_re_im(re: Self::Real, im: Self::Real) -> Self;

    /// Magnitude (Euclidean norm) as a real.
    /// Annotation:
    /// - Purpose: Executes `abs_real` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn abs_real(self) -> Self::Real;

    /// Squared magnitude as a real.
    /// Annotation:
    /// - Purpose: Executes `norm_sqr_real` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn norm_sqr_real(self) -> Self::Real;

    /// Square root with type preserved.
    /// Annotation:
    /// - Purpose: Executes `sqrt` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn sqrt(self) -> Self;

    /// Finite check (integers are always finite).
    /// Annotation:
    /// - Purpose: Checks whether `finite` condition is true.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn is_finite(self) -> bool;
}

/**
Serialization-capable scalar extension trait.

This extends compute `Scalar` with serde capabilities, while keeping numeric
kernel bounds minimal.
*/
pub trait ScalarSerde: Scalar + Serialize + DeserializeOwned {}

impl<T> ScalarSerde for T where T: Scalar + Serialize + DeserializeOwned {}

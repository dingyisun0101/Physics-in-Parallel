//! Core scalar traits shared by real and complex implementations.
//!
//! # Design contract
//!
//! `Scalar` is the crate's user-facing "number-like value" abstraction. It is
//! intentionally broader than a minimal algebraic trait because PiP is meant to
//! feel natural to physicists and mathematicians working with arrays, vectors,
//! grids, and simulation state. Real primitives, integer primitives, and complex
//! values all live behind one concept.
//!
//! The API is organized around one important distinction:
//!
//! 1. Type-preserving operations return `Self`.
//!    These operations keep computation inside the same scalar category:
//!    `T -> T`. For example, `sqrt`, `conj`, `abs`, and `norm_sqr` all return
//!    the same scalar type they received. Integer implementations therefore use
//!    crate-defined integer semantics such as integer square roots and saturating
//!    squared norms where needed.
//!
//! 2. Projection and construction operations cross the real/complex boundary.
//!    `re`, `im`, `abs_real`, and `norm_sqr_real` intentionally return
//!    `Self::Real`, while `from_re_im` constructs `Self` from real-domain
//!    components. These methods are the explicit boundary used by casting and
//!    interoperability code.
//!
//! This split keeps everyday scalar math convenient while making type
//! conversion points visible. For real scalar targets, `from_re_im` ignores the
//! imaginary component by design; fallible high-level casts can choose to reject
//! nonzero imaginary parts if a stricter workflow needs that behavior.

use core::any::type_name;
use core::fmt::{Debug, Display};
use core::iter::{Product, Sum};

use num_traits::{Num, NumCast, One, Zero};

/// Internal sealing so only this crate can implement scalar traits.
pub(crate) mod scalar_sealed {
    /// Internal bulk dispatch; overrides never alter public scalar semantics.
    pub trait Sealed: Sized {
        fn binary_into(
            left: &[Self],
            right: &[Self],
            output: &mut [Self],
            op: crate::math::kernels::BinaryOp,
        ) where
            Self: super::Scalar,
        {
            for ((out, &a), &b) in output.iter_mut().zip(left).zip(right) {
                *out = op.apply(a, b);
            }
        }

        fn scale_slice(values: &mut [Self], scalar: Self)
        where
            Self: super::Scalar,
        {
            for value in values {
                *value = *value * scalar;
            }
        }
    }
}

use scalar_sealed::Sealed;

/// Error returned when one scalar cannot be represented as another scalar type.
///
/// Conversion happens in the real-domain boundary:
///	- project source value into `(re, im)`;
///	- cast each component into the target scalar's `Real` type;
///	- reconstruct target value with `U::from_re_im(re, im)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarCastError {
    /// The source real component cannot be represented in the target real type.
    RealPartOutOfRange {
        source: &'static str,
        target: &'static str,
    },
    /// The source imaginary component cannot be represented in the target real type.
    ImagPartOutOfRange {
        source: &'static str,
        target: &'static str,
    },
}

impl core::fmt::Display for ScalarCastError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RealPartOutOfRange { source, target } => write!(
                f,
                "real component of scalar type {source} cannot be represented as {target}"
            ),
            Self::ImagPartOutOfRange { source, target } => write!(
                f,
                "imaginary component of scalar type {source} cannot be represented as {target}"
            ),
        }
    }
}

impl std::error::Error for ScalarCastError {}

/**
    Compute-oriented scalar trait for real, and complex numeric types.

    `Real` is the scalar's associated real-domain type:
    - `Self` for real and integer scalars.
    - `f32`/`f64` for complex scalars.

    The trait is sealed so PiP can keep one coherent scalar contract across the
    crate while the API is still evolving.
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
        + Product<Self::Real>
        + Scalar<Real = Self::Real>;

    // ---------------------------------------------------------------------
    // Type-preserving scalar operations: Self -> Self
    // ---------------------------------------------------------------------

    /// Complex conjugate (identity for reals).
    /// Purpose:
    ///	Return the complex conjugate while preserving the scalar type; real and
    ///	integer implementations return themselves unchanged.
    /// Parameters:
    ///	- (none): Operates on the scalar value passed as `self`.
    fn conj(self) -> Self;

    /// Magnitude represented back in the same scalar category.
    ///
    /// For real and integer scalars this is the ordinary absolute value under
    /// that implementation's overflow policy. For complex scalars this returns
    /// a complex value with magnitude in the real component and zero imaginary
    /// component.
    #[inline]
    fn abs(self) -> Self {
        Self::from_re_im(self.abs_real(), Self::Real::zero())
    }

    /// Squared magnitude represented back in the same scalar category.
    ///
    /// This is the type-preserving companion to `norm_sqr_real`.
    #[inline]
    fn norm_sqr(self) -> Self {
        Self::from_re_im(self.norm_sqr_real(), Self::Real::zero())
    }

    /// Square root with type preserved.
    /// Purpose:
    ///	Compute the square root using this scalar category's own semantics,
    ///	returning the same scalar type rather than automatically widening.
    /// Parameters:
    ///	- (none): Operates on the scalar value passed as `self`.
    fn sqrt(self) -> Self;

    // ---------------------------------------------------------------------
    // Real-domain projections and scalar casts
    // ---------------------------------------------------------------------

    /// Real part (`self` for reals).
    /// Purpose:
    ///	Project the scalar into its real component so generic code can inspect
    ///	real and complex values through one interface.
    /// Parameters:
    ///	- (none): Operates on the scalar value passed as `self`.
    fn re(self) -> Self::Real;

    /// Imaginary part (`0` for reals).
    /// Purpose:
    ///	Project the scalar into its imaginary component; real and integer
    ///	scalars return zero.
    /// Parameters:
    ///	- (none): Operates on the scalar value passed as `self`.
    fn im(self) -> Self::Real;

    /// Magnitude (Euclidean norm) as a real.
    /// Purpose:
    ///	Return the scalar magnitude in the associated real domain, making the
    ///	real-valued projection explicit for complex inputs.
    /// Parameters:
    ///	- (none): Operates on the scalar value passed as `self`.
    fn abs_real(self) -> Self::Real;

    /// Squared magnitude as a real.
    /// Purpose:
    ///	Return the squared magnitude in the associated real domain; this is the
    ///	projection counterpart to the type-preserving `norm_sqr`.
    /// Parameters:
    ///	- (none): Operates on the scalar value passed as `self`.
    fn norm_sqr_real(self) -> Self::Real;

    /// Fallibly cast this scalar into another scalar type.
    ///
    /// This is the universal scalar conversion operation. It is intentionally
    /// located at the real-domain boundary rather than in the type-preserving
    /// section because the target scalar type may differ from the source type.
    ///
    /// This is a range-checked conversion, not a lossless conversion: floating
    /// precision may be rounded and fractional parts truncated by integer targets.
    ///
    /// Semantics:
    ///	- `Self` is projected to `(self.re(), self.im())`.
    ///	- Each component is cast into `U::Real` with `NumCast`.
    ///	- `U::from_re_im` constructs the target scalar.
    ///
    /// Consequences:
    ///	- integer-to-float and float-to-integer casts use `num_traits` casting
    ///	  rules and fail when the target cannot represent the value;
    ///	- complex-to-complex casts preserve both components when representable;
    ///	- complex-to-real casts use the same rule as `from_re_im` for real
    ///	  targets, so the imaginary component is ignored after it is verified
    ///	  representable in the target real type.
    #[inline]
    fn try_cast<U: Scalar>(self) -> Result<U, ScalarCastError> {
        let source = type_name::<Self>();
        let target = type_name::<U>();

        let source_re = self.re();
        let source_im = self.im();

        let re = <U::Real as NumCast>::from(source_re)
            .ok_or(ScalarCastError::RealPartOutOfRange { source, target })?;
        if source_re.is_finite() && !re.is_finite() {
            return Err(ScalarCastError::RealPartOutOfRange { source, target });
        }

        let im = <U::Real as NumCast>::from(source_im)
            .ok_or(ScalarCastError::ImagPartOutOfRange { source, target })?;
        if source_im.is_finite() && !im.is_finite() {
            return Err(ScalarCastError::ImagPartOutOfRange { source, target });
        }

        Ok(U::from_re_im(re, im))
    }

    /// Cast this scalar into another scalar type, panicking if conversion fails.
    ///
    /// Use `try_cast` when out-of-range values are expected input data.
    #[inline]
    fn cast<U: Scalar>(self) -> U {
        self.try_cast::<U>().expect("scalar cast failed")
    }

    // ---------------------------------------------------------------------
    // Real-domain construction: Self::Real x Self::Real -> Self
    // ---------------------------------------------------------------------

    /// Construct from real/imaginary parts.
    ///
    /// Purpose:
    ///	Build `Self` from real-domain components.
    /// Parameters:
    ///	- `re`: real component.
    ///	- `im`: imaginary component. Real scalar implementations ignore this by design.
    fn from_re_im(re: Self::Real, im: Self::Real) -> Self;

    // ---------------------------------------------------------------------
    // Value classification
    // ---------------------------------------------------------------------

    /// Finite check (integers are always finite).
    /// Purpose:
    ///	Report whether the scalar is finite under its numeric category; integer
    ///	scalars are always finite, while floating and complex scalars inspect
    ///	their floating components.
    /// Parameters:
    ///	- (none): Operates on the scalar value passed as `self`.
    fn is_finite(self) -> bool;
}

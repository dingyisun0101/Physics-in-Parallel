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

    /// Annotation:
    /// - Purpose: Executes `conj` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn conj(self) -> Self { Complex::conj(&self) }
    /// Annotation:
    /// - Purpose: Executes `re` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn re(self) -> Self::Real { self.re }
    /// Annotation:
    /// - Purpose: Executes `im` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn im(self) -> Self::Real { self.im }
    /// Annotation:
    /// - Purpose: Builds this value from `re_im` input.
    /// - Parameters:
    ///   - `re` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
    ///   - `im` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
    #[inline] fn from_re_im(re: Self::Real, im: Self::Real) -> Self { Complex::new(re, im) }

    /// Annotation:
    /// - Purpose: Executes `abs_real` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn abs_real(self) -> Self::Real { self.norm() }
    /// Annotation:
    /// - Purpose: Executes `norm_sqr_real` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn norm_sqr_real(self) -> Self::Real { self.norm_sqr() }
    /// Annotation:
    /// - Purpose: Executes `sqrt` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn sqrt(self) -> Self { Complex::sqrt(self) }
    /// Annotation:
    /// - Purpose: Checks whether `finite` condition is true.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn is_finite(self) -> bool { self.re.is_finite() && self.im.is_finite() }
}

impl Scalar for Complex<f64> {
    type Real = f64;

    /// Annotation:
    /// - Purpose: Executes `conj` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn conj(self) -> Self { Complex::conj(&self) }
    /// Annotation:
    /// - Purpose: Executes `re` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn re(self) -> Self::Real { self.re }
    /// Annotation:
    /// - Purpose: Executes `im` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn im(self) -> Self::Real { self.im }
    /// Annotation:
    /// - Purpose: Builds this value from `re_im` input.
    /// - Parameters:
    ///   - `re` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
    ///   - `im` (`Self::Real`): Parameter of type `Self::Real` used by `from_re_im`.
    #[inline] fn from_re_im(re: Self::Real, im: Self::Real) -> Self { Complex::new(re, im) }

    /// Annotation:
    /// - Purpose: Executes `abs_real` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn abs_real(self) -> Self::Real { self.norm() }
    /// Annotation:
    /// - Purpose: Executes `norm_sqr_real` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn norm_sqr_real(self) -> Self::Real { self.norm_sqr() }
    /// Annotation:
    /// - Purpose: Executes `sqrt` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn sqrt(self) -> Self { Complex::sqrt(self) }
    /// Annotation:
    /// - Purpose: Checks whether `finite` condition is true.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn is_finite(self) -> bool { self.re.is_finite() && self.im.is_finite() }
}


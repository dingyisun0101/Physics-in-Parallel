/*!
Unified ndarray conversion trait for math containers.

This trait intentionally mirrors the project-wide naming convention:
- `from_ndarray`
- `to_ndarray`
*/

pub trait NdarrayConvert: Sized {
    /// Concrete ndarray representation used by this container.
    type NdArray;

    /// Construct `Self` from an ndarray value.
    /// Annotation:
    /// - Purpose: Builds this value from `ndarray` input.
    /// - Parameters:
    ///   - `array` (`&Self::NdArray`): ndarray input used for conversion/interoperability.
    fn from_ndarray(array: &Self::NdArray) -> Self;

    /// Convert `Self` into its ndarray representation.
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn to_ndarray(&self) -> Self::NdArray;
}


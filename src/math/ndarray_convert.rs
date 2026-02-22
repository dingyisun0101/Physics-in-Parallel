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
    fn from_ndarray(array: &Self::NdArray) -> Self;

    /// Convert `Self` into its ndarray representation.
    fn to_ndarray(&self) -> Self::NdArray;
}


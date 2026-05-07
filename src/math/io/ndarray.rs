/*!
Unified ndarray conversion trait for IO/interoperability.

This trait intentionally mirrors the project-wide naming convention:
- `from_ndarray`
- `to_ndarray`
*/

pub trait NdarrayConvert: Sized {
    /// Concrete ndarray representation used by this container.
    type NdArray;

    /// Construct `Self` from an ndarray value.
    /// Details:
    /// - Purpose: Converts an owned or borrowed ndarray representation into
    ///   the corresponding PiP math container while preserving shape and
    ///   row-major logical element order.
    /// - Parameters:
    ///   - `array` (`&Self::NdArray`): ndarray value whose dimensions and
    ///     scalar entries should be imported.
    fn from_ndarray(array: &Self::NdArray) -> Self;

    /// Convert `Self` into its ndarray representation.
    /// Details:
    /// - Purpose: Exports a PiP math container into an ndarray value with the
    ///   same logical shape and scalar entries.
    /// - Parameters:
    ///   - (none): Reads the source container without modifying it.
    fn to_ndarray(&self) -> Self::NdArray;
}

use ndarray::{ArrayD, IxDyn};

use crate::math::scalar::Scalar;
use crate::math::tensor::rank_n::{
    Dense, Sparse, Tensor, dense::Tensor as DenseStorage, sparse::Tensor as SparseStorage,
    tensor_trait::TensorTrait,
};

impl<T: Scalar> DenseStorage<T> {
    pub(crate) fn from_ndarray(array: &ArrayD<T>) -> Self {
        let owned = array.to_owned();
        let shape = owned.shape().to_vec();
        let (data, _) = owned.into_raw_vec_and_offset();
        Self::from_parts_unchecked(shape, data)
    }

    pub(crate) fn to_ndarray(&self) -> ArrayD<T> {
        ArrayD::from_shape_vec(IxDyn(self.shape()), self.data().to_vec())
            .expect("Tensor::to_ndarray: shape/data length mismatch")
    }
}

impl<T: Scalar> SparseStorage<T> {
    pub(crate) fn from_ndarray_storage(array: &ArrayD<T>) -> Self {
        let dense = DenseStorage::<T>::from_ndarray(array);
        Self::from_dense(&dense)
    }

    pub(crate) fn to_ndarray_storage(&self) -> ArrayD<T> {
        self.to_dense().to_ndarray()
    }
}

impl<T: Scalar> NdarrayConvert for Tensor<T, Dense> {
    type NdArray = ArrayD<T>;

    fn from_ndarray(array: &Self::NdArray) -> Self {
        Tensor::<T, Dense>::from_storage(DenseStorage::<T>::from_ndarray(array))
    }

    fn to_ndarray(&self) -> Self::NdArray {
        self.storage().to_ndarray()
    }
}

impl<T: Scalar> NdarrayConvert for Tensor<T, Sparse> {
    type NdArray = ArrayD<T>;

    fn from_ndarray(array: &Self::NdArray) -> Self {
        Tensor::<T, Sparse>::from_storage(SparseStorage::<T>::from_ndarray_storage(array))
    }

    fn to_ndarray(&self) -> Self::NdArray {
        self.storage().to_ndarray_storage()
    }
}

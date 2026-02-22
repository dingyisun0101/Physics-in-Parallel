/*!
Unified front API for tensor storage backends.

`Tensor<T, B>` provides one user-facing type parameterized by backend marker `B`:
- `Dense`  -> contiguous dense storage
- `Sparse` -> hash-backed sparse storage

This facade delegates to existing backend implementations in `core::dense` and
`core::sparse` while exposing a consistent construction/access surface.
*/

use core::marker::PhantomData;

use ndarray::ArrayD;

use crate::math::{
    ndarray_convert::NdarrayConvert,
    scalar::Scalar,
};

use super::{
    dense::Tensor as DenseStorage,
    sparse::Tensor as SparseStorage,
    tensor_trait::TensorTrait,
};

/// Dense backend marker for `Tensor<T, B>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dense;

/// Sparse backend marker for `Tensor<T, B>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sparse;

/// Backend type mapping for unified `Tensor`.
pub trait Backend<T: Scalar> {
    type Storage: TensorTrait<T>;
}

impl<T: Scalar> Backend<T> for Dense {
    type Storage = DenseStorage<T>;
}

impl<T: Scalar> Backend<T> for Sparse {
    type Storage = SparseStorage<T>;
}

/// Unified tensor facade.
#[derive(Debug, Clone)]
pub struct Tensor<T: Scalar, B: Backend<T> = Dense> {
    inner: B::Storage,
    _backend: PhantomData<B>,
}

impl<T: Scalar, B: Backend<T>> Tensor<T, B> {
    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `storage` input.
    /// - Parameters:
    ///   - `inner` (`B::Storage`): Parameter of type `B::Storage` used by `from_storage`.
    pub fn from_storage(inner: B::Storage) -> Self {
        Self { inner, _backend: PhantomData }
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `into_storage` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn into_storage(self) -> B::Storage {
        self.inner
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `storage` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn storage(&self) -> &B::Storage {
        &self.inner
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `storage_mut` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn storage_mut(&mut self) -> &mut B::Storage {
        &mut self.inner
    }
}

impl<T: Scalar, B: Backend<T>> Tensor<T, B>
where
    B::Storage: TensorTrait<T>,
{
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `empty` logic for this module.
    /// - Parameters:
    ///   - `shape` (`&[usize]`): Shape metadata defining tensor/grid dimensions.
    pub fn empty(shape: &[usize]) -> Self {
        Self::from_storage(<B::Storage as TensorTrait<T>>::empty(shape))
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Returns the logical shape metadata.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Returns the `sum` value.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn get_sum(&self) -> T {
        self.inner.get_sum()
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `get` logic for this module.
    /// - Parameters:
    ///   - `idx` (`&[isize]`): Index argument selecting an element or slot.
    pub fn get(&self, idx: &[isize]) -> T
    where
        T: Copy,
    {
        self.inner.get(idx)
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `set` logic for this module.
    /// - Parameters:
    ///   - `idx` (`&[isize]`): Index argument selecting an element or slot.
    ///   - `val` (`T`): Value provided by caller for write/update behavior.
    pub fn set(&mut self, idx: &[isize], val: T) {
        self.inner.set(idx, val);
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `par_fill` logic for this module.
    /// - Parameters:
    ///   - `value` (`T`): Value provided by caller for write/update behavior.
    pub fn par_fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.inner.par_fill(value);
    }

    #[inline]
    pub fn par_map_in_place<F>(&mut self, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send,
    {
        self.inner.par_map_in_place(f);
    }

    #[inline]
    pub fn par_zip_with_inplace<F>(&mut self, other: &Self, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T, T) -> T + Sync + Send,
    {
        self.inner.par_zip_with_inplace(&other.inner, f);
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Prints a human-readable representation.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn print(&self) {
        self.inner.print();
    }
}

impl<T: Scalar> Tensor<T, Dense> {
    #[inline]
    pub fn cast_to<U: Scalar + Send + Sync>(&self) -> Tensor<U, Dense> {
        Tensor::<U, Dense>::from_storage(self.inner.cast_to::<U>())
    }

    #[inline]
    pub fn try_cast_to<U: Scalar>(&self) -> Result<Tensor<U, Dense>, &'static str> {
        self.inner
            .try_cast_to::<U>()
            .map(Tensor::<U, Dense>::from_storage)
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `sparse` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_sparse(&self) -> Tensor<T, Sparse> {
        Tensor::<T, Sparse>::from_storage(self.inner.to_sparse())
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `sparse` input.
    /// - Parameters:
    ///   - `s` (`&Tensor<T, Sparse>`): Parameter of type `&Tensor<T, Sparse>` used by `from_sparse`.
    pub fn from_sparse(s: &Tensor<T, Sparse>) -> Self {
        Self::from_storage(DenseStorage::<T>::from_sparse(&s.inner))
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `ndarray` input.
    /// - Parameters:
    ///   - `array` (`&ArrayD<T>`): ndarray input used for conversion/interoperability.
    pub fn from_ndarray(array: &ArrayD<T>) -> Self {
        Self::from_storage(DenseStorage::<T>::from_ndarray(array))
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_ndarray(&self) -> ArrayD<T> {
        self.inner.to_ndarray()
    }
}

impl<T: Scalar> Tensor<T, Sparse> {
    #[inline]
    pub fn cast_to<U: Scalar + Send + Sync>(&self) -> Tensor<U, Sparse> {
        Tensor::<U, Sparse>::from_storage(self.inner.cast_to::<U>())
    }

    #[inline]
    pub fn try_cast_to<U: Scalar>(&self) -> Result<Tensor<U, Sparse>, &'static str> {
        self.inner
            .try_cast_to::<U>()
            .map(Tensor::<U, Sparse>::from_storage)
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `nnz` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Returns the current length/size.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn len_dense(&self) -> usize {
        self.inner.len_dense()
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `dense` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_dense(&self) -> Tensor<T, Dense> {
        Tensor::<T, Dense>::from_storage(self.inner.to_dense())
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `dense` input.
    /// - Parameters:
    ///   - `d` (`&Tensor<T, Dense>`): Parameter of type `&Tensor<T, Dense>` used by `from_dense`.
    pub fn from_dense(d: &Tensor<T, Dense>) -> Self {
        Self::from_storage(SparseStorage::<T>::from_dense(&d.inner))
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `ndarray` input.
    /// - Parameters:
    ///   - `array` (`&ArrayD<T>`): ndarray input used for conversion/interoperability.
    pub fn from_ndarray(array: &ArrayD<T>) -> Self {
        Self::from_storage(SparseStorage::<T>::from_ndarray(array))
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_ndarray(&self) -> ArrayD<T> {
        self.inner.to_ndarray()
    }
}

impl<T: Scalar> NdarrayConvert for Tensor<T, Dense> {
    type NdArray = ArrayD<T>;

    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `ndarray` input.
    /// - Parameters:
    ///   - `array` (`&Self::NdArray`): ndarray input used for conversion/interoperability.
    fn from_ndarray(array: &Self::NdArray) -> Self {
        Tensor::<T, Dense>::from_ndarray(array)
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn to_ndarray(&self) -> Self::NdArray {
        Tensor::<T, Dense>::to_ndarray(self)
    }
}

impl<T: Scalar> NdarrayConvert for Tensor<T, Sparse> {
    type NdArray = ArrayD<T>;

    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `ndarray` input.
    /// - Parameters:
    ///   - `array` (`&Self::NdArray`): ndarray input used for conversion/interoperability.
    fn from_ndarray(array: &Self::NdArray) -> Self {
        Tensor::<T, Sparse>::from_ndarray(array)
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn to_ndarray(&self) -> Self::NdArray {
        Tensor::<T, Sparse>::to_ndarray(self)
    }
}


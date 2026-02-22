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
    pub fn from_storage(inner: B::Storage) -> Self {
        Self { inner, _backend: PhantomData }
    }

    #[inline]
    pub fn into_storage(self) -> B::Storage {
        self.inner
    }

    #[inline]
    pub fn storage(&self) -> &B::Storage {
        &self.inner
    }

    #[inline]
    pub fn storage_mut(&mut self) -> &mut B::Storage {
        &mut self.inner
    }
}

impl<T: Scalar, B: Backend<T>> Tensor<T, B>
where
    B::Storage: TensorTrait<T>,
{
    #[inline]
    pub fn empty(shape: &[usize]) -> Self {
        Self::from_storage(<B::Storage as TensorTrait<T>>::empty(shape))
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    #[inline]
    pub fn get_sum(&self) -> T {
        self.inner.get_sum()
    }

    #[inline]
    pub fn get(&self, idx: &[isize]) -> T
    where
        T: Copy,
    {
        self.inner.get(idx)
    }

    #[inline]
    pub fn set(&mut self, idx: &[isize], val: T) {
        self.inner.set(idx, val);
    }

    #[inline]
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
    pub fn to_sparse(&self) -> Tensor<T, Sparse> {
        Tensor::<T, Sparse>::from_storage(self.inner.to_sparse())
    }

    #[inline]
    pub fn from_sparse(s: &Tensor<T, Sparse>) -> Self {
        Self::from_storage(DenseStorage::<T>::from_sparse(&s.inner))
    }

    #[inline]
    pub fn from_ndarray(array: &ArrayD<T>) -> Self {
        Self::from_storage(DenseStorage::<T>::from_ndarray(array))
    }

    #[inline]
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
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    #[inline]
    pub fn len_dense(&self) -> usize {
        self.inner.len_dense()
    }

    #[inline]
    pub fn to_dense(&self) -> Tensor<T, Dense> {
        Tensor::<T, Dense>::from_storage(self.inner.to_dense())
    }

    #[inline]
    pub fn from_dense(d: &Tensor<T, Dense>) -> Self {
        Self::from_storage(SparseStorage::<T>::from_dense(&d.inner))
    }

    #[inline]
    pub fn from_ndarray(array: &ArrayD<T>) -> Self {
        Self::from_storage(SparseStorage::<T>::from_ndarray(array))
    }

    #[inline]
    pub fn to_ndarray(&self) -> ArrayD<T> {
        self.inner.to_ndarray()
    }
}

impl<T: Scalar> NdarrayConvert for Tensor<T, Dense> {
    type NdArray = ArrayD<T>;

    #[inline]
    fn from_ndarray(array: &Self::NdArray) -> Self {
        Tensor::<T, Dense>::from_ndarray(array)
    }

    #[inline]
    fn to_ndarray(&self) -> Self::NdArray {
        Tensor::<T, Dense>::to_ndarray(self)
    }
}

impl<T: Scalar> NdarrayConvert for Tensor<T, Sparse> {
    type NdArray = ArrayD<T>;

    #[inline]
    fn from_ndarray(array: &Self::NdArray) -> Self {
        Tensor::<T, Sparse>::from_ndarray(array)
    }

    #[inline]
    fn to_ndarray(&self) -> Self::NdArray {
        Tensor::<T, Sparse>::to_ndarray(self)
    }
}


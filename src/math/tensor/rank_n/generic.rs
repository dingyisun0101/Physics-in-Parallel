/*!
Generic front API for tensor storage backends.

`Tensor<T, B>` provides one user-facing type parameterized by backend marker `B`:
- `Dense`  -> contiguous dense storage
- `Sparse` -> hash-backed sparse storage

This facade delegates to existing backend implementations in `rank_n::dense` and
`rank_n::sparse` while exposing a consistent construction/access surface. The
element type is always constrained by `T: Scalar`, so tensor algorithms operate
on PiP scalar values rather than backend-specific primitive assumptions.
*/

use core::marker::PhantomData;

use crate::math::scalar::{Scalar, ScalarCastError};

use super::{
    dense::Tensor as DenseStorage, sparse::Tensor as SparseStorage, tensor_trait::TensorTrait,
};

/// Dense backend marker for `Tensor<T, B>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dense;

/// Sparse backend marker for `Tensor<T, B>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sparse;

/// Backend type mapping for generic `Tensor`.
pub trait Backend<T: Scalar> {
    type Storage: TensorTrait<T>;
}

impl<T: Scalar> Backend<T> for Dense {
    type Storage = DenseStorage<T>;
}

impl<T: Scalar> Backend<T> for Sparse {
    type Storage = SparseStorage<T>;
}

/// Generic tensor facade.
#[derive(Debug, Clone)]
pub struct Tensor<T: Scalar, B: Backend<T> = Dense> {
    inner: B::Storage,
    _backend: PhantomData<B>,
}

impl<T: Scalar, B: Backend<T>> Tensor<T, B> {
    #[inline]
    /// Details:
    /// - Purpose: Builds this value from `storage` input.
    /// - Parameters:
    ///   - `inner` (`B::Storage`): Parameter of type `B::Storage` used by `from_storage`.
    pub(crate) fn from_storage(inner: B::Storage) -> Self {
        Self {
            inner,
            _backend: PhantomData,
        }
    }

    #[inline]
    /// Details:
    /// - Purpose: Executes `storage` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub(crate) fn storage(&self) -> &B::Storage {
        &self.inner
    }
}

impl<T: Scalar, B: Backend<T>> Tensor<T, B>
where
    B::Storage: TensorTrait<T>,
{
    #[inline]
    /// Details:
    /// - Purpose: Executes `empty` logic for this module.
    /// - Parameters:
    ///   - `shape` (`&[usize]`): Shape metadata defining tensor/grid dimensions.
    pub fn empty(shape: &[usize]) -> Self {
        Self::from_storage(<B::Storage as TensorTrait<T>>::empty(shape))
    }

    #[inline]
    pub fn zeros(shape: &[usize]) -> Self {
        Self::empty(shape)
    }

    #[inline]
    /// Details:
    /// - Purpose: Returns the logical shape metadata.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    #[inline]
    /// Tensor rank.
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[inline]
    /// Logical dense size.
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    #[inline]
    /// Details:
    /// - Purpose: Returns the `sum` value.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn get_sum(&self) -> T {
        self.inner.get_sum()
    }

    #[inline]
    /// Sum of tensor values.
    pub fn sum(&self) -> T {
        self.inner.sum()
    }

    #[inline]
    /// Details:
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
    pub fn get_mut(&mut self, idx: &[isize]) -> &mut T {
        self.inner.get_mut(idx)
    }

    #[inline]
    /// Details:
    /// - Purpose: Executes `set` logic for this module.
    /// - Parameters:
    ///   - `idx` (`&[isize]`): Index argument selecting an element or slot.
    ///   - `val` (`T`): Value provided by caller for write/update behavior.
    pub fn set(&mut self, idx: &[isize], val: T) {
        self.inner.set(idx, val);
    }

    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.inner.fill(value);
    }

    #[inline]
    pub fn map<F>(&self, f: F) -> Self
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send,
    {
        Self::from_storage(self.inner.map(f))
    }

    #[inline]
    pub fn map_in_place<F>(&mut self, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send,
    {
        self.inner.map_in_place(f);
    }

    #[inline]
    pub fn zip_with<RhsBackend, F>(&self, other: &Tensor<T, RhsBackend>, f: F) -> Self
    where
        RhsBackend: Backend<T>,
        RhsBackend::Storage: TensorTrait<T>,
        T: Copy + Send + Sync,
        F: Fn(T, T) -> T + Sync + Send,
    {
        Self::from_storage(self.inner.zip_with(&other.inner, f))
    }

    #[inline]
    pub fn conj(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.conj())
    }

    #[inline]
    pub fn abs(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.abs())
    }

    #[inline]
    pub fn norm_sqr(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.norm_sqr())
    }

    #[inline]
    pub fn sqrt(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.sqrt())
    }

    #[inline]
    pub fn scalar_mul(&self, scalar: T) -> Self
    where
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.scalar_mul(scalar))
    }

    #[inline]
    pub fn elem_mul<RhsBackend>(&self, other: &Tensor<T, RhsBackend>) -> Self
    where
        RhsBackend: Backend<T>,
        RhsBackend::Storage: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.elem_mul(&other.inner))
    }

    #[inline]
    pub fn elem_div<RhsBackend>(&self, other: &Tensor<T, RhsBackend>) -> Self
    where
        RhsBackend: Backend<T>,
        RhsBackend::Storage: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.elem_div(&other.inner))
    }

    #[inline]
    pub fn transpose(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.transpose())
    }

    #[inline]
    pub fn hermitian_transpose(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.hermitian_transpose())
    }

    #[inline]
    pub fn dot<RhsBackend>(&self, other: &Tensor<T, RhsBackend>) -> T
    where
        RhsBackend: Backend<T>,
        RhsBackend::Storage: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        self.inner.dot(&other.inner)
    }

    #[inline]
    pub fn hermitian_dot<RhsBackend>(&self, other: &Tensor<T, RhsBackend>) -> T
    where
        RhsBackend: Backend<T>,
        RhsBackend::Storage: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        self.inner.hermitian_dot(&other.inner)
    }

    #[inline]
    pub fn norm_sqr_real(&self) -> T::Real
    where
        T: Copy + Send + Sync,
        T::Real: Send + Sync,
    {
        self.inner.norm_sqr_real()
    }

    #[inline]
    pub fn norm(&self) -> T
    where
        T: Copy + Send + Sync,
        T::Real: Send + Sync,
    {
        self.inner.norm()
    }

    #[inline]
    pub fn cross<RhsBackend>(&self, other: &Tensor<T, RhsBackend>) -> Self
    where
        RhsBackend: Backend<T>,
        RhsBackend::Storage: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.cross(&other.inner))
    }

    #[inline]
    pub fn wedge<RhsBackend>(&self, other: &Tensor<T, RhsBackend>) -> Self
    where
        RhsBackend: Backend<T>,
        RhsBackend::Storage: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.wedge(&other.inner))
    }

    #[inline]
    pub fn matmul<RhsBackend>(&self, other: &Tensor<T, RhsBackend>) -> Self
    where
        RhsBackend: Backend<T>,
        RhsBackend::Storage: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        Self::from_storage(self.inner.matmul(&other.inner))
    }

    #[inline]
    /// Details:
    /// - Purpose: Prints a human-readable representation.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn print(&self) {
        self.inner.print();
    }
}

impl<T: Scalar> Tensor<T, Dense> {
    #[inline]
    pub fn from_vec(shape: &[usize], data: Vec<T>) -> Self {
        Self::from_storage(DenseStorage::<T>::from_vec(shape, data))
    }

    pub fn from_fn<F>(shape: &[usize], mut f: F) -> Self
    where
        F: FnMut(&[isize]) -> T,
    {
        let mut out = Self::empty(shape);
        let rank = shape.len();
        for k in 0..out.size() {
            let mut rem = k;
            let mut idx = vec![0isize; rank];
            for axis in (0..rank).rev() {
                idx[axis] = (rem % shape[axis]) as isize;
                rem /= shape[axis];
            }
            out.set(&idx, f(&idx));
        }
        out
    }

    #[inline]
    pub fn cast_to<U: Scalar + Send + Sync>(&self) -> Tensor<U, Dense> {
        Tensor::<U, Dense>::from_storage(self.inner.cast_to::<U>())
    }

    #[inline]
    pub fn try_cast_to<U: Scalar>(&self) -> Result<Tensor<U, Dense>, ScalarCastError> {
        self.inner
            .try_cast_to::<U>()
            .map(Tensor::<U, Dense>::from_storage)
    }

    #[inline]
    /// Details:
    /// - Purpose: Converts this value into `sparse` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_sparse(&self) -> Tensor<T, Sparse> {
        Tensor::<T, Sparse>::from_storage(self.inner.to_sparse())
    }

    #[inline]
    /// Details:
    /// - Purpose: Builds this value from `sparse` input.
    /// - Parameters:
    ///   - `s` (`&Tensor<T, Sparse>`): Parameter of type `&Tensor<T, Sparse>` used by `from_sparse`.
    pub fn from_sparse(s: &Tensor<T, Sparse>) -> Self {
        Self::from_storage(DenseStorage::<T>::from_sparse(&s.inner))
    }
}

impl<T: Scalar> Tensor<T, Sparse> {
    #[inline]
    pub fn from_triplets(
        shape: Vec<usize>,
        triplets: impl IntoIterator<Item = (Vec<usize>, T)>,
    ) -> Self {
        Self::from_storage(SparseStorage::<T>::from_triplets(shape, triplets))
    }

    #[inline]
    pub fn cast_to<U: Scalar + Send + Sync>(&self) -> Tensor<U, Sparse> {
        Tensor::<U, Sparse>::from_storage(self.inner.cast_to::<U>())
    }

    #[inline]
    pub fn try_cast_to<U: Scalar>(&self) -> Result<Tensor<U, Sparse>, ScalarCastError> {
        self.inner
            .try_cast_to::<U>()
            .map(Tensor::<U, Sparse>::from_storage)
    }

    #[inline]
    /// Details:
    /// - Purpose: Executes `nnz` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    #[inline]
    /// Details:
    /// - Purpose: Returns the current length/size.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn len_dense(&self) -> usize {
        self.inner.len_dense()
    }

    #[inline]
    /// Details:
    /// - Purpose: Converts this value into `dense` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_dense(&self) -> Tensor<T, Dense> {
        Tensor::<T, Dense>::from_storage(self.inner.to_dense())
    }

    #[inline]
    /// Details:
    /// - Purpose: Builds this value from `dense` input.
    /// - Parameters:
    ///   - `d` (`&Tensor<T, Dense>`): Parameter of type `&Tensor<T, Dense>` used by `from_dense`.
    pub fn from_dense(d: &Tensor<T, Dense>) -> Self {
        Self::from_storage(SparseStorage::<T>::from_dense(&d.inner))
    }
}

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

use rayon::prelude::*;

use crate::math::scalar::{Scalar, ScalarCastError};
use crate::threading::{parallel_chunk_len, should_parallelize_elements};

use super::{
    dense::Tensor as DenseStorage, sparse::Tensor as SparseStorage, tensor_trait::TensorTrait,
};

/// Dense backend marker for `Tensor<T, B>`.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct Dense;

/// Sparse backend marker for `Tensor<T, B>`.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T: Scalar, B: Backend<T> = Dense> {
    inner: B::Storage,
    _backend: PhantomData<B>,
}

impl<T: Scalar, B: Backend<T>> Tensor<T, B> {
    #[inline]
    /// Details:
    /// - Purpose: Wraps a backend-specific dense or sparse storage object in
    ///   the public `Tensor<T, B>` facade while preserving the backend marker.
    /// - Parameters:
    ///   - `inner` (`B::Storage`): Already-validated backend storage to expose
    ///     through the generic tensor API.
    pub(crate) fn from_storage(inner: B::Storage) -> Self {
        Self {
            inner,
            _backend: PhantomData,
        }
    }

    #[inline]
    /// Details:
    /// - Purpose: Gives internal IO and conversion code read-only access to
    ///   the backend storage without making storage details part of the public
    ///   end-user API.
    /// - Parameters:
    ///   - (none): Borrows the facade's backend storage.
    pub(crate) fn storage(&self) -> &B::Storage {
        &self.inner
    }

    #[inline]
    pub(crate) fn storage_mut(&mut self) -> &mut B::Storage {
        &mut self.inner
    }
}

impl<T: Scalar, B: Backend<T>> Tensor<T, B>
where
    B::Storage: TensorTrait<T>,
{
    #[inline]
    /// Details:
    /// - Purpose: Creates a tensor of the selected backend with the requested
    ///   shape and that backend's zero/default empty representation.
    /// - Parameters:
    ///   - `shape` (`&[usize]`): Non-empty list of positive axis lengths.
    pub fn empty(shape: &[usize]) -> Self {
        Self::from_storage(<B::Storage as TensorTrait<T>>::empty(shape))
    }

    #[inline]
    pub fn zeros(shape: &[usize]) -> Self {
        Self::empty(shape)
    }

    #[inline]
    /// Details:
    /// - Purpose: Returns the logical axis lengths shared by every backend
    ///   representation of this tensor.
    /// - Parameters:
    ///   - (none): Delegates to the backend storage shape.
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

    /// Converts a periodic multidimensional coordinate into a canonical flat index.
    #[inline]
    pub fn flat_index(&self, coordinate: &[isize]) -> usize {
        self.inner.index(coordinate)
    }

    #[inline]
    /// Sum of tensor values.
    pub fn sum(&self) -> T {
        self.inner.sum()
    }

    #[inline]
    /// Details:
    /// - Purpose: Reads a scalar at a multidimensional coordinate through the
    ///   backend's indexing rules; dense returns stored values and sparse
    ///   returns zero for implicit entries.
    /// - Parameters:
    ///   - `idx` (`&[isize]`): One signed coordinate per tensor axis.
    pub fn get(&self, idx: &[isize]) -> T
    where
        T: Copy,
    {
        self.inner.get(idx)
    }

    /// Reads through a periodically normalized row-major flat index.
    #[inline]
    pub fn get_flat(&self, index: isize) -> T
    where
        T: Copy,
    {
        self.inner.get_flat(index)
    }

    #[inline]
    pub fn get_mut(&mut self, idx: &[isize]) -> &mut T {
        self.inner.get_mut(idx)
    }

    /// Mutably borrows through a periodically normalized flat index.
    #[inline]
    pub fn get_flat_mut(&mut self, index: isize) -> &mut T {
        self.inner.get_flat_mut(index)
    }

    #[inline]
    /// Details:
    /// - Purpose: Writes a scalar at a multidimensional coordinate through the
    ///   backend's update rules; sparse zero writes remove explicit storage.
    /// - Parameters:
    ///   - `idx` (`&[isize]`): One signed coordinate per tensor axis.
    ///   - `val` (`T`): Scalar value to store at that logical coordinate.
    pub fn set(&mut self, idx: &[isize], val: T) {
        self.inner.set(idx, val);
    }

    /// Writes through a periodically normalized flat index.
    #[inline]
    pub fn set_flat(&mut self, index: isize, value: T) {
        self.inner.set_flat(index, value);
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
}

impl<T: Scalar> Tensor<T, Dense> {
    /// Borrows the complete dense row-major backing buffer without copying.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.data()
    }

    /// Mutably borrows the complete dense row-major backing buffer without copying.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.data_mut()
    }

    #[inline]
    pub(crate) fn data(&self) -> &[T] {
        self.as_slice()
    }

    /// Constructs a dense tensor after validating shape and row-major data length.
    pub fn try_from_vec(shape: &[usize], data: Vec<T>) -> super::TensorResult<Self> {
        let expected = super::errors::checked_num_elements(shape)?;
        if data.len() != expected {
            return Err(super::TensorError::DataLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self::from_storage(DenseStorage::<T>::from_parts_unchecked(
            shape.to_vec(),
            data,
        )))
    }

    /// Constructs a dense tensor, panicking when shape and data disagree.
    #[inline]
    pub fn from_vec(shape: &[usize], data: Vec<T>) -> Self {
        Self::try_from_vec(shape, data).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Copies values from another identically shaped dense tensor.
    pub fn copy_from(&mut self, source: &Self) -> super::TensorResult<()>
    where
        T: Copy,
    {
        if self.shape() != source.shape() {
            return Err(super::TensorError::ShapeMismatch {
                lhs: self.shape().to_vec(),
                rhs: source.shape().to_vec(),
            });
        }
        self.as_mut_slice().copy_from_slice(source.as_slice());
        Ok(())
    }

    /// Writes an allocation-free elementwise zip of two tensors into `self`.
    pub fn zip_from<F>(&mut self, lhs: &Self, rhs: &Self, function: F) -> super::TensorResult<()>
    where
        T: Copy + Send + Sync,
        F: Fn(T, T) -> T + Send + Sync,
    {
        if self.shape() != lhs.shape() {
            return Err(super::TensorError::ShapeMismatch {
                lhs: self.shape().to_vec(),
                rhs: lhs.shape().to_vec(),
            });
        }
        if self.shape() != rhs.shape() {
            return Err(super::TensorError::ShapeMismatch {
                lhs: self.shape().to_vec(),
                rhs: rhs.shape().to_vec(),
            });
        }
        let lhs = lhs.as_slice();
        let rhs = rhs.as_slice();
        let output = self.as_mut_slice();
        if !should_parallelize_elements(output.len()) {
            output
                .iter_mut()
                .zip(lhs)
                .zip(rhs)
                .for_each(|((target, left), right)| *target = function(*left, *right));
            return Ok(());
        }
        let minimum = parallel_chunk_len(output.len()).unwrap_or(1);
        output
            .par_iter_mut()
            .zip(lhs.par_iter())
            .zip(rhs.par_iter())
            .with_min_len(minimum)
            .for_each(|((target, left), right)| *target = function(*left, *right));
        Ok(())
    }

    /// Visits fixed-width row-major chunks serially or through bounded Rayon work.
    pub fn for_each_chunk_mut<F>(&mut self, width: usize, function: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Send + Sync,
    {
        assert!(width > 0, "tensor chunk width must be positive");
        assert!(
            self.size().is_multiple_of(width),
            "tensor size must be divisible by chunk width"
        );
        let values = self.as_mut_slice();
        let chunks = values.len() / width;
        if chunks < 2 || !should_parallelize_elements(values.len()) {
            values
                .chunks_exact_mut(width)
                .enumerate()
                .for_each(|(index, chunk)| function(index, chunk));
            return;
        }
        values
            .par_chunks_exact_mut(width)
            .with_min_len(parallel_chunk_len(chunks).unwrap_or(1))
            .enumerate()
            .for_each(|(index, chunk)| function(index, chunk));
    }

    /// Sums values in stable row-major order without a parallel reduction tree.
    pub fn sum_serial(&self) -> T
    where
        T: Copy,
    {
        self.as_slice()
            .iter()
            .copied()
            .fold(T::zero(), |sum, value| sum + value)
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
    /// - Purpose: Converts a dense facade tensor into a sparse facade tensor by
    ///   preserving shape and explicitly storing only nonzero entries.
    /// - Parameters:
    ///   - (none): Reads this dense tensor.
    pub fn to_sparse(&self) -> Tensor<T, Sparse> {
        Tensor::<T, Sparse>::from_storage(self.inner.to_sparse())
    }

    #[inline]
    /// Details:
    /// - Purpose: Builds a dense facade tensor by materializing every implicit
    ///   sparse zero into a full dense buffer.
    /// - Parameters:
    ///   - `s` (`&Tensor<T, Sparse>`): Sparse facade tensor to densify.
    pub fn from_sparse(s: &Tensor<T, Sparse>) -> Self {
        Self::from_storage(DenseStorage::<T>::from_sparse(&s.inner))
    }
}

impl<T> Eq for Tensor<T, Dense> where T: Scalar + Eq {}

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
    /// - Purpose: Returns how many entries are explicitly stored by the sparse
    ///   backend; implicit zeros are excluded.
    /// - Parameters:
    ///   - (none): Reads the sparse backend's storage map length.
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    #[inline]
    /// Details:
    /// - Purpose: Returns the number of logical positions in the sparse tensor's
    ///   dense domain, equal to the product of shape axes.
    /// - Parameters:
    ///   - (none): Uses the sparse tensor's shape metadata.
    pub fn len_dense(&self) -> usize {
        self.inner.len_dense()
    }

    #[inline]
    /// Details:
    /// - Purpose: Converts a sparse facade tensor into a dense facade tensor by
    ///   allocating the full dense buffer and filling implicit zeros.
    /// - Parameters:
    ///   - (none): Reads this sparse tensor.
    pub fn to_dense(&self) -> Tensor<T, Dense> {
        Tensor::<T, Dense>::from_storage(self.inner.to_dense())
    }

    #[inline]
    /// Details:
    /// - Purpose: Builds a sparse facade tensor by scanning a dense tensor and
    ///   keeping only entries that are not `T::zero()`.
    /// - Parameters:
    ///   - `d` (`&Tensor<T, Dense>`): Dense facade tensor to compress.
    pub fn from_dense(d: &Tensor<T, Dense>) -> Self {
        Self::from_storage(SparseStorage::<T>::from_dense(&d.inner))
    }
}

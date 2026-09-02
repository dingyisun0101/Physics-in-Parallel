//! Backend-agnostic tensor API.

use core::fmt;

use ahash::AHashSet;
use num_traits::Zero;
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::math::scalar::{Scalar, ScalarCastError};

use super::rank_n::errors::{
    TensorError, TensorResult, checked_num_elements, ensure_coordinate_rank, ensure_same_shape,
};
use super::rank_n::{Dense, Sparse, Tensor as BackendTensor};

/// Storage representation used by a backend-agnostic tensor.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageKind {
    Dense,
    Sparse,
}

#[derive(Clone, Debug, PartialEq)]
enum Storage<T: Scalar> {
    Dense(BackendTensor<T, Dense>),
    Sparse(BackendTensor<T, Sparse>),
}

/// An N-dimensional tensor with either dense or sparse private storage.
///
/// Dense and sparse tensors share this one public type and mathematical API.
/// Constructors choose the initial representation, and operations never
/// change it implicitly.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T: Scalar> {
    storage: Storage<T>,
}

impl<T> Eq for Tensor<T> where T: Scalar + Eq {}

impl<T: Scalar> Tensor<T> {
    /// Constructs a dense tensor from row-major logical values.
    pub fn from_dense(shape: &[usize], values: Vec<T>) -> TensorResult<Self> {
        let expected = checked_num_elements(shape)?;
        if values.len() != expected {
            return Err(TensorError::DataLengthMismatch {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            storage: Storage::Dense(BackendTensor::<T, Dense>::from_storage(
                super::rank_n::dense::Tensor::from_parts_unchecked(shape.to_vec(), values),
            )),
        })
    }

    /// Constructs a sparse tensor from strict logical coordinates.
    ///
    /// Coordinates must be in bounds and unique. Explicit zeros are accepted
    /// as input but are not stored.
    pub fn from_sparse_entries(
        shape: &[usize],
        entries: impl IntoIterator<Item = (Vec<usize>, T)>,
    ) -> TensorResult<Self> {
        let size = checked_num_elements(shape)?;
        let mut seen = AHashSet::default();
        let mut flat_entries = Vec::new();
        for (coordinates, value) in entries {
            let flat = checked_flat_index(shape, &coordinates)?;
            if !seen.insert(flat) {
                return Err(TensorError::DuplicateCoordinate { coordinates });
            }
            if value != T::zero() {
                flat_entries.push((flat, value));
            }
        }
        debug_assert!(flat_entries.iter().all(|(index, _)| *index < size));
        Ok(Self::from_sparse_flat_entries_unchecked(
            shape.to_vec(),
            flat_entries,
        ))
    }

    /// Constructs an all-zero sparse tensor.
    pub fn zeros(shape: &[usize]) -> TensorResult<Self> {
        checked_num_elements(shape)?;
        Ok(Self::from_sparse_flat_entries_unchecked(
            shape.to_vec(),
            Vec::new(),
        ))
    }

    /// Constructs a dense tensor whose logical elements all equal `value`.
    pub fn filled(shape: &[usize], value: T) -> TensorResult<Self> {
        let size = checked_num_elements(shape)?;
        Self::from_dense(shape, vec![value; size])
    }

    /// Constructs a dense tensor by evaluating coordinates in row-major order.
    pub fn from_fn<F>(shape: &[usize], mut function: F) -> TensorResult<Self>
    where
        F: FnMut(&[usize]) -> T,
    {
        let size = checked_num_elements(shape)?;
        let mut coordinates = vec![0; shape.len()];
        let mut values = Vec::with_capacity(size);
        for index in 0..size {
            coordinates_from_flat(shape, index, &mut coordinates);
            values.push(function(&coordinates));
        }
        Self::from_dense(shape, values)
    }

    /// Constructs a sparse identity matrix.
    pub fn identity(size: usize) -> TensorResult<Self> {
        let shape = [size, size];
        checked_num_elements(&shape)?;
        Self::from_sparse_entries(
            &shape,
            (0..size).map(|coordinate| (vec![coordinate, coordinate], T::one())),
        )
    }

    /// Returns the selected storage representation.
    pub const fn storage_kind(&self) -> StorageKind {
        match self.storage {
            Storage::Dense(_) => StorageKind::Dense,
            Storage::Sparse(_) => StorageKind::Sparse,
        }
    }

    /// Converts this tensor to dense storage when necessary.
    pub fn make_dense(&mut self) {
        if let Storage::Sparse(storage) = &self.storage {
            self.storage = Storage::Dense(storage.to_dense());
        }
    }

    /// Converts this tensor to sparse storage when necessary.
    pub fn make_sparse(&mut self) {
        if let Storage::Dense(storage) = &self.storage {
            self.storage = Storage::Sparse(storage.to_sparse());
        }
    }

    /// Returns the logical axis extents.
    pub fn shape(&self) -> &[usize] {
        match &self.storage {
            Storage::Dense(storage) => storage.shape(),
            Storage::Sparse(storage) => storage.shape(),
        }
    }

    /// Returns the number of logical axes.
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Returns the total logical element count.
    pub fn size(&self) -> usize {
        match &self.storage {
            Storage::Dense(storage) => storage.size(),
            Storage::Sparse(storage) => storage.size(),
        }
    }

    /// Reads one logical value at strict multidimensional coordinates.
    pub fn get(&self, coordinates: &[usize]) -> TensorResult<T> {
        let index = checked_flat_index(self.shape(), coordinates)?;
        Ok(self.get_flat_unchecked(index))
    }

    /// Writes one logical value at strict multidimensional coordinates.
    pub fn set(&mut self, coordinates: &[usize], value: T) -> TensorResult<()> {
        let index = checked_flat_index(self.shape(), coordinates)?;
        self.set_flat_unchecked(index, value);
        Ok(())
    }

    /// Traverses every logical value in row-major coordinate order.
    ///
    /// # Complexity
    ///
    /// This is `O(total logical elements)` for both dense and sparse storage.
    /// Use advanced stored-entry access when sparse support traversal is
    /// intended.
    pub fn values(&self) -> Values<'_, T> {
        Values {
            tensor: self,
            next: 0,
        }
    }

    /// Returns the sum of every logical value.
    pub fn sum(&self) -> T {
        self.values().fold(T::zero(), |sum, value| sum + value)
    }

    /// Replaces every logical value while retaining the current storage kind.
    ///
    /// # Complexity
    ///
    /// A nonzero fill of sparse storage writes every logical element and can
    /// require dense-scale memory.
    pub fn fill(&mut self, value: T) {
        let values = vec![value; self.size()];
        self.replace_with_values(values);
    }

    /// Maps every logical value while retaining the receiver's storage kind.
    ///
    /// # Result storage
    ///
    /// The result has the same storage kind as `self`.
    ///
    /// # Complexity
    ///
    /// Mapping is `O(total logical elements)` for both representations. A map
    /// for which `f(0)` is nonzero can produce dense occupancy in sparse
    /// storage.
    pub fn map<F>(&self, function: F) -> Self
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let values = self.values().map(function).collect();
        Self::from_values_unchecked(self.shape(), self.storage_kind(), values)
    }

    /// Maps every logical value in place without changing storage kind.
    pub fn map_in_place<F>(&mut self, function: F)
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let values = self.values().map(function).collect();
        self.replace_with_values(values);
    }

    /// Adds two equally shaped tensors.
    pub fn add(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip(rhs, |left, right| left + right)
    }

    /// Subtracts two equally shaped tensors.
    pub fn subtract(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip(rhs, |left, right| left - right)
    }

    /// Multiplies two equally shaped tensors element by element.
    pub fn multiply(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip(rhs, |left, right| left * right)
    }

    /// Divides two equally shaped tensors element by element.
    pub fn divide(&self, rhs: &Self) -> TensorResult<Self> {
        self.zip(rhs, |left, right| left / right)
    }

    /// Writes elementwise addition into an existing output tensor.
    pub fn add_into(&self, rhs: &Self, output: &mut Self) -> TensorResult<()> {
        self.zip_into(rhs, output, |left, right| left + right)
    }

    /// Writes elementwise subtraction into an existing output tensor.
    pub fn subtract_into(&self, rhs: &Self, output: &mut Self) -> TensorResult<()> {
        self.zip_into(rhs, output, |left, right| left - right)
    }

    /// Writes elementwise multiplication into an existing output tensor.
    pub fn multiply_into(&self, rhs: &Self, output: &mut Self) -> TensorResult<()> {
        self.zip_into(rhs, output, |left, right| left * right)
    }

    /// Writes elementwise division into an existing output tensor.
    pub fn divide_into(&self, rhs: &Self, output: &mut Self) -> TensorResult<()> {
        self.zip_into(rhs, output, |left, right| left / right)
    }

    /// Multiplies every logical value by a scalar.
    pub fn scale(&self, scalar: T) -> Self {
        self.map(|value| value * scalar)
    }

    /// Returns the elementwise complex conjugate.
    pub fn conjugate(&self) -> Self {
        self.map(Scalar::conj)
    }

    /// Returns the elementwise absolute value in the same scalar category.
    pub fn abs(&self) -> Self {
        self.map(Scalar::abs)
    }

    /// Returns the elementwise squared norm in the same scalar category.
    pub fn norm_squared(&self) -> Self {
        self.map(Scalar::norm_sqr)
    }

    /// Returns the elementwise square root.
    pub fn sqrt(&self) -> Self {
        self.map(Scalar::sqrt)
    }

    /// Returns the rank-two transpose while preserving storage kind.
    pub fn transpose(&self) -> TensorResult<Self> {
        self.ensure_rank("transpose", 2)?;
        let rows = self.shape()[0];
        let columns = self.shape()[1];
        let values = (0..columns * rows)
            .map(|index| {
                let row = index / rows;
                let column = index % rows;
                self.get_flat_unchecked(column * columns + row)
            })
            .collect();
        Ok(Self::from_values_unchecked(
            &[columns, rows],
            self.storage_kind(),
            values,
        ))
    }

    /// Returns the rank-two Hermitian transpose while preserving storage kind.
    pub fn hermitian_transpose(&self) -> TensorResult<Self> {
        Ok(self.transpose()?.conjugate())
    }

    /// Computes a dot product over equally shaped logical values.
    pub fn dot(&self, rhs: &Self) -> TensorResult<T> {
        ensure_same_shape(self.shape(), rhs.shape())?;
        Ok((0..self.size())
            .into_par_iter()
            .map(|index| self.get_flat_unchecked(index) * rhs.get_flat_unchecked(index))
            .reduce(T::zero, |left, right| left + right))
    }

    /// Computes a Hermitian dot product over equally shaped logical values.
    pub fn hermitian_dot(&self, rhs: &Self) -> TensorResult<T> {
        ensure_same_shape(self.shape(), rhs.shape())?;
        Ok((0..self.size())
            .into_par_iter()
            .map(|index| self.get_flat_unchecked(index).conj() * rhs.get_flat_unchecked(index))
            .reduce(T::zero, |left, right| left + right))
    }

    /// Returns the real-valued squared Euclidean norm.
    pub fn norm_squared_real(&self) -> T::Real {
        self.values()
            .map(Scalar::norm_sqr_real)
            .fold(T::Real::zero(), |left, right| left + right)
    }

    /// Returns the Euclidean norm in the tensor scalar category.
    pub fn norm(&self) -> T {
        T::from_re_im(self.norm_squared_real().sqrt(), T::Real::zero())
    }

    /// Computes the cross product of two length-three rank-one tensors.
    pub fn cross(&self, rhs: &Self) -> TensorResult<Self> {
        self.ensure_vector_len("cross", 3)?;
        rhs.ensure_vector_len("cross", 3)?;
        let left = [
            self.get_flat_unchecked(0),
            self.get_flat_unchecked(1),
            self.get_flat_unchecked(2),
        ];
        let right = [
            rhs.get_flat_unchecked(0),
            rhs.get_flat_unchecked(1),
            rhs.get_flat_unchecked(2),
        ];
        Ok(Self::from_values_unchecked(
            &[3],
            self.storage_kind(),
            vec![
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            ],
        ))
    }

    /// Computes the exterior product of two equally sized rank-one tensors.
    pub fn wedge(&self, rhs: &Self) -> TensorResult<Self> {
        self.ensure_rank("wedge", 1)?;
        rhs.ensure_rank("wedge", 1)?;
        if self.shape()[0] != rhs.shape()[0] {
            return Err(TensorError::DimensionMismatch {
                operation: "wedge",
                lhs: self.shape()[0],
                rhs: rhs.shape()[0],
            });
        }
        let size = self.shape()[0];
        let values = (0..size * size)
            .map(|index| {
                let row = index / size;
                let column = index % size;
                self.get_flat_unchecked(row) * rhs.get_flat_unchecked(column)
                    - self.get_flat_unchecked(column) * rhs.get_flat_unchecked(row)
            })
            .collect();
        Ok(Self::from_values_unchecked(
            &[size, size],
            self.storage_kind(),
            values,
        ))
    }

    /// Multiplies two rank-two tensors.
    ///
    /// # Result storage
    ///
    /// The result has the same storage kind as `self`.
    ///
    /// # Complexity
    ///
    /// The general kernel performs `O(mkn)` scalar operations. Preserving a
    /// sparse receiver can create dense occupancy.
    pub fn matmul(&self, rhs: &Self) -> TensorResult<Self> {
        self.ensure_rank("matmul", 2)?;
        rhs.ensure_rank("matmul", 2)?;
        let rows = self.shape()[0];
        let inner = self.shape()[1];
        if inner != rhs.shape()[0] {
            return Err(TensorError::DimensionMismatch {
                operation: "matmul",
                lhs: inner,
                rhs: rhs.shape()[0],
            });
        }
        let columns = rhs.shape()[1];
        let values = (0..rows * columns)
            .into_par_iter()
            .map(|index| {
                let row = index / columns;
                let column = index % columns;
                (0..inner).fold(T::zero(), |sum, offset| {
                    sum + self.get_flat_unchecked(row * inner + offset)
                        * rhs.get_flat_unchecked(offset * columns + column)
                })
            })
            .collect();
        Ok(Self::from_values_unchecked(
            &[rows, columns],
            self.storage_kind(),
            values,
        ))
    }

    /// Casts every logical value while preserving storage kind.
    pub fn cast<U: Scalar>(&self) -> Result<Tensor<U>, ScalarCastError> {
        let values = self
            .values()
            .map(|value| value.try_cast::<U>())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Tensor::<U>::from_values_unchecked(
            self.shape(),
            self.storage_kind(),
            values,
        ))
    }

    fn zip<F>(&self, rhs: &Self, function: F) -> TensorResult<Self>
    where
        F: Fn(T, T) -> T + Send + Sync,
    {
        ensure_same_shape(self.shape(), rhs.shape())?;
        let values = (0..self.size())
            .into_par_iter()
            .map(|index| {
                function(
                    self.get_flat_unchecked(index),
                    rhs.get_flat_unchecked(index),
                )
            })
            .collect();
        Ok(Self::from_values_unchecked(
            self.shape(),
            self.storage_kind(),
            values,
        ))
    }

    fn zip_into<F>(&self, rhs: &Self, output: &mut Self, function: F) -> TensorResult<()>
    where
        F: Fn(T, T) -> T + Send + Sync,
    {
        ensure_same_shape(self.shape(), rhs.shape())?;
        ensure_same_shape(self.shape(), output.shape())?;
        let values = (0..self.size())
            .into_par_iter()
            .map(|index| {
                function(
                    self.get_flat_unchecked(index),
                    rhs.get_flat_unchecked(index),
                )
            })
            .collect();
        output.replace_with_values(values);
        Ok(())
    }

    fn ensure_rank(&self, operation: &'static str, expected: usize) -> TensorResult<()> {
        if self.rank() != expected {
            return Err(TensorError::ExpectedRank {
                operation,
                expected,
                actual: self.rank(),
            });
        }
        Ok(())
    }

    fn ensure_vector_len(&self, operation: &'static str, expected: usize) -> TensorResult<()> {
        self.ensure_rank(operation, 1)?;
        if self.shape()[0] != expected {
            return Err(TensorError::DimensionMismatch {
                operation,
                lhs: self.shape()[0],
                rhs: expected,
            });
        }
        Ok(())
    }

    fn replace_with_values(&mut self, values: Vec<T>) {
        self.storage =
            Self::storage_from_values_unchecked(self.shape(), self.storage_kind(), values);
    }

    fn from_values_unchecked(shape: &[usize], kind: StorageKind, values: Vec<T>) -> Self {
        Self {
            storage: Self::storage_from_values_unchecked(shape, kind, values),
        }
    }

    fn storage_from_values_unchecked(
        shape: &[usize],
        kind: StorageKind,
        values: Vec<T>,
    ) -> Storage<T> {
        match kind {
            StorageKind::Dense => Storage::Dense(BackendTensor::<T, Dense>::from_storage(
                super::rank_n::dense::Tensor::from_parts_unchecked(shape.to_vec(), values),
            )),
            StorageKind::Sparse => {
                let entries = values
                    .into_iter()
                    .enumerate()
                    .filter(|(_, value)| *value != T::zero())
                    .collect();
                Storage::Sparse(BackendTensor::<T, Sparse>::from_storage(
                    super::rank_n::sparse::Tensor::from_flat_pairs(shape.to_vec(), entries),
                ))
            }
        }
    }

    pub(crate) fn from_sparse_flat_entries_unchecked(
        shape: Vec<usize>,
        entries: Vec<(usize, T)>,
    ) -> Self {
        Self {
            storage: Storage::Sparse(BackendTensor::<T, Sparse>::from_storage(
                super::rank_n::sparse::Tensor::from_flat_pairs(shape, entries),
            )),
        }
    }

    pub(crate) fn get_flat_unchecked(&self, index: usize) -> T {
        debug_assert!(index < self.size());
        match &self.storage {
            Storage::Dense(storage) => storage.get_flat(index as isize),
            Storage::Sparse(storage) => storage.get_flat(index as isize),
        }
    }

    pub(crate) fn set_flat_unchecked(&mut self, index: usize, value: T) {
        debug_assert!(index < self.size());
        match &mut self.storage {
            Storage::Dense(storage) => storage.set_flat(index as isize, value),
            Storage::Sparse(storage) => storage.set_flat(index as isize, value),
        }
    }

    pub(crate) fn dense_values(&self) -> Option<&[T]> {
        match &self.storage {
            Storage::Dense(storage) => Some(storage.storage().data()),
            Storage::Sparse(_) => None,
        }
    }

    pub(crate) fn dense_values_mut(&mut self) -> Option<&mut [T]> {
        match &mut self.storage {
            Storage::Dense(storage) => Some(storage.storage_mut().data_mut()),
            Storage::Sparse(_) => None,
        }
    }

    pub(crate) fn sparse_entries(&self) -> Option<impl Iterator<Item = (usize, T)> + '_> {
        match &self.storage {
            Storage::Dense(_) => None,
            Storage::Sparse(storage) => Some(
                storage
                    .storage()
                    .iter()
                    .map(|(&index, &value)| (index, value)),
            ),
        }
    }
}

/// Logical row-major tensor value iterator.
pub struct Values<'a, T: Scalar> {
    tensor: &'a Tensor<T>,
    next: usize,
}

impl<T: Scalar> Iterator for Values<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.tensor.size() {
            return None;
        }
        let value = self.tensor.get_flat_unchecked(self.next);
        self.next += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.tensor.size() - self.next;
        (remaining, Some(remaining))
    }
}

impl<T: Scalar> ExactSizeIterator for Values<'_, T> {}

fn checked_flat_index(shape: &[usize], coordinates: &[usize]) -> TensorResult<usize> {
    ensure_coordinate_rank(shape, coordinates.len())?;
    let mut index = 0;
    for (axis, (&coordinate, &extent)) in coordinates.iter().zip(shape).enumerate() {
        if coordinate >= extent {
            return Err(TensorError::CoordinateOutOfBounds {
                axis,
                coordinate,
                extent,
            });
        }
        index = index * extent + coordinate;
    }
    Ok(index)
}

fn coordinates_from_flat(shape: &[usize], mut index: usize, coordinates: &mut [usize]) {
    for axis in (0..shape.len()).rev() {
        coordinates[axis] = index % shape[axis];
        index /= shape[axis];
    }
}

#[derive(Serialize)]
#[serde(tag = "storage", content = "tensor", rename_all = "snake_case")]
enum StorageRef<'a, T: Scalar + Serialize> {
    Dense(&'a BackendTensor<T, Dense>),
    Sparse(&'a BackendTensor<T, Sparse>),
}

#[derive(Deserialize)]
#[serde(tag = "storage", content = "tensor", rename_all = "snake_case")]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
enum StorageOwned<T: Scalar> {
    Dense(BackendTensor<T, Dense>),
    Sparse(BackendTensor<T, Sparse>),
}

impl<T> Serialize for Tensor<T>
where
    T: Scalar + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self.storage {
            Storage::Dense(storage) => StorageRef::Dense(storage).serialize(serializer),
            Storage::Sparse(storage) => StorageRef::Sparse(storage).serialize(serializer),
        }
    }
}

impl<'de, T> Deserialize<'de> for Tensor<T>
where
    T: Scalar + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let storage = match StorageOwned::<T>::deserialize(deserializer)? {
            StorageOwned::Dense(storage) => Storage::Dense(storage),
            StorageOwned::Sparse(storage) => Storage::Sparse(storage),
        };
        Ok(Self { storage })
    }
}

impl fmt::Display for StorageKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense => formatter.write_str("dense"),
            Self::Sparse => formatter.write_str("sparse"),
        }
    }
}

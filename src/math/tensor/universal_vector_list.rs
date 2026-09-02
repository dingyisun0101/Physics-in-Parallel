//! Backend-agnostic fixed-width vector-list API.

use core::fmt;

use num_traits::Zero;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::math::scalar::{Scalar, ScalarCastError};

use super::rank_2::vector_list::generic::DynVectorList;
use super::rank_n::errors::TensorError;
use super::universal::{StorageKind, Tensor, Values};

/// Failure while constructing or operating on a vector list.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VectorListError {
    Tensor(TensorError),
    VectorLength { expected: usize, actual: usize },
    AxisLength { expected: usize, actual: usize },
    ScaleLength { expected: usize, actual: usize },
}

impl From<TensorError> for VectorListError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl fmt::Display for VectorListError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(error) => write!(formatter, "invalid vector list: {error}"),
            Self::VectorLength { expected, actual } => write!(
                formatter,
                "vector component count mismatch: expected {expected}, got {actual}"
            ),
            Self::AxisLength { expected, actual } => write!(
                formatter,
                "vector-list axis length mismatch: expected {expected}, got {actual}"
            ),
            Self::ScaleLength { expected, actual } => write!(
                formatter,
                "vector-list scale count mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for VectorListError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

/// A list of fixed-width vectors with private dense or sparse storage.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorList<T: Scalar> {
    tensor: Tensor<T>,
}

impl<T> Eq for VectorList<T> where T: Scalar + Eq {}

impl<T: Scalar> VectorList<T> {
    /// Constructs dense row-major vectors.
    pub fn from_dense(
        dimensions: usize,
        num_vectors: usize,
        values: Vec<T>,
    ) -> Result<Self, VectorListError> {
        Self::from_tensor(Tensor::from_dense(&[num_vectors, dimensions], values)?)
    }

    /// Constructs sparse vectors from `(vector, component, value)` entries.
    pub fn from_sparse_entries(
        dimensions: usize,
        num_vectors: usize,
        entries: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> Result<Self, VectorListError> {
        let entries = entries
            .into_iter()
            .map(|(vector, component, value)| (vec![vector, component], value));
        Self::from_tensor(Tensor::from_sparse_entries(
            &[num_vectors, dimensions],
            entries,
        )?)
    }

    /// Constructs all-zero sparse vectors.
    pub fn zeros(dimensions: usize, num_vectors: usize) -> Result<Self, VectorListError> {
        Self::from_tensor(Tensor::zeros(&[num_vectors, dimensions])?)
    }

    /// Constructs dense vectors filled with one value.
    pub fn filled(
        dimensions: usize,
        num_vectors: usize,
        value: T,
    ) -> Result<Self, VectorListError> {
        Self::from_tensor(Tensor::filled(&[num_vectors, dimensions], value)?)
    }

    /// Constructs dense vectors from a coordinate function.
    pub fn from_fn<F>(
        dimensions: usize,
        num_vectors: usize,
        mut function: F,
    ) -> Result<Self, VectorListError>
    where
        F: FnMut(usize, usize) -> T,
    {
        Self::from_tensor(Tensor::from_fn(
            &[num_vectors, dimensions],
            |coordinates| function(coordinates[0], coordinates[1]),
        )?)
    }

    pub const fn storage_kind(&self) -> StorageKind {
        self.tensor.storage_kind()
    }

    pub fn make_dense(&mut self) {
        self.tensor.make_dense();
    }

    pub fn make_sparse(&mut self) {
        self.tensor.make_sparse();
    }

    pub fn dim(&self) -> usize {
        self.tensor.shape()[1]
    }

    pub fn num_vectors(&self) -> usize {
        self.tensor.shape()[0]
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.num_vectors(), self.dim()]
    }

    pub fn get(&self, vector: usize, component: usize) -> Result<T, VectorListError> {
        Ok(self.tensor.get(&[vector, component])?)
    }

    pub fn set(
        &mut self,
        vector: usize,
        component: usize,
        value: T,
    ) -> Result<(), VectorListError> {
        Ok(self.tensor.set(&[vector, component], value)?)
    }

    pub fn vector(&self, vector: usize) -> Result<Vec<T>, VectorListError> {
        if vector >= self.num_vectors() {
            return Err(TensorError::CoordinateOutOfBounds {
                axis: 0,
                coordinate: vector,
                extent: self.num_vectors(),
            }
            .into());
        }
        let start = vector * self.dim();
        Ok((start..start + self.dim())
            .map(|index| self.tensor.get_flat_unchecked(index))
            .collect())
    }

    pub fn set_vector(&mut self, vector: usize, values: &[T]) -> Result<(), VectorListError> {
        if values.len() != self.dim() {
            return Err(VectorListError::VectorLength {
                expected: self.dim(),
                actual: values.len(),
            });
        }
        for (component, &value) in values.iter().enumerate() {
            self.set(vector, component, value)?;
        }
        Ok(())
    }

    pub fn axis(&self, component: usize) -> Result<Vec<T>, VectorListError> {
        if component >= self.dim() {
            return Err(TensorError::CoordinateOutOfBounds {
                axis: 1,
                coordinate: component,
                extent: self.dim(),
            }
            .into());
        }
        Ok((0..self.num_vectors())
            .map(|vector| {
                self.tensor
                    .get_flat_unchecked(vector * self.dim() + component)
            })
            .collect())
    }

    pub fn set_axis(&mut self, component: usize, values: &[T]) -> Result<(), VectorListError> {
        if values.len() != self.num_vectors() {
            return Err(VectorListError::AxisLength {
                expected: self.num_vectors(),
                actual: values.len(),
            });
        }
        for (vector, &value) in values.iter().enumerate() {
            self.set(vector, component, value)?;
        }
        Ok(())
    }

    /// Traverses every logical component in vector-major order.
    ///
    /// # Complexity
    ///
    /// This is `O(num_vectors * dimensions)` for dense and sparse storage.
    pub fn values(&self) -> Values<'_, T> {
        self.tensor.values()
    }

    pub fn fill(&mut self, value: T) {
        self.tensor.fill(value);
    }

    pub fn scale_vectors(&mut self, scales: &[T]) -> Result<(), VectorListError> {
        if scales.len() != self.num_vectors() {
            return Err(VectorListError::ScaleLength {
                expected: self.num_vectors(),
                actual: scales.len(),
            });
        }
        let mut values = self.values().collect::<Vec<_>>();
        for (vector, row) in values.chunks_exact_mut(self.dim()).enumerate() {
            for value in row {
                *value = *value * scales[vector];
            }
        }
        self.replace_values(values);
        Ok(())
    }

    pub fn norms(&self) -> Vec<T> {
        self.values()
            .collect::<Vec<_>>()
            .chunks_exact(self.dim())
            .map(|row| {
                row.iter()
                    .copied()
                    .map(Scalar::norm_sqr)
                    .fold(T::zero(), |left, right| left + right)
                    .sqrt()
            })
            .collect()
    }

    pub fn norms_real(&self) -> Vec<T::Real> {
        self.values()
            .collect::<Vec<_>>()
            .chunks_exact(self.dim())
            .map(|row| {
                row.iter()
                    .copied()
                    .map(Scalar::norm_sqr_real)
                    .fold(T::Real::zero(), |left, right| left + right)
                    .sqrt()
            })
            .collect()
    }

    pub fn normalize(&mut self) -> Result<(), VectorListError> {
        let scales = self
            .norms()
            .into_iter()
            .map(|norm| {
                if norm == T::zero() {
                    T::one()
                } else {
                    T::one() / norm
                }
            })
            .collect::<Vec<_>>();
        self.scale_vectors(&scales)
    }

    pub fn to_polar(&self) -> Result<(Vec<T>, Self), VectorListError> {
        let norms = self.norms();
        let scales = norms
            .iter()
            .copied()
            .map(|norm| {
                if norm == T::zero() {
                    T::zero()
                } else {
                    T::one() / norm
                }
            })
            .collect::<Vec<_>>();
        let mut units = self.clone();
        units.scale_vectors(&scales)?;
        Ok((norms, units))
    }

    /// Adds two vector lists and preserves the receiver's storage kind.
    ///
    /// # Result storage
    ///
    /// The result has the same storage kind as `self`.
    pub fn add(&self, rhs: &Self) -> Result<Self, VectorListError> {
        Self::from_tensor(self.tensor.add(&rhs.tensor)?)
    }

    pub fn subtract(&self, rhs: &Self) -> Result<Self, VectorListError> {
        Self::from_tensor(self.tensor.subtract(&rhs.tensor)?)
    }

    pub fn multiply(&self, rhs: &Self) -> Result<Self, VectorListError> {
        Self::from_tensor(self.tensor.multiply(&rhs.tensor)?)
    }

    pub fn divide(&self, rhs: &Self) -> Result<Self, VectorListError> {
        Self::from_tensor(self.tensor.divide(&rhs.tensor)?)
    }

    pub fn scale(&self, scalar: T) -> Self {
        Self {
            tensor: self.tensor.scale(scalar),
        }
    }

    pub fn cast<U: Scalar>(&self) -> Result<VectorList<U>, ScalarCastError> {
        Ok(VectorList {
            tensor: self.tensor.cast()?,
        })
    }

    pub(crate) fn replace_values(&mut self, values: Vec<T>) {
        let kind = self.storage_kind();
        self.tensor = match kind {
            StorageKind::Dense => Tensor::from_dense(self.tensor.shape(), values)
                .expect("existing vector-list shape is valid"),
            StorageKind::Sparse => Tensor::from_sparse_entries(
                self.tensor.shape(),
                values
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| (vec![index / self.dim(), index % self.dim()], value)),
            )
            .expect("existing vector-list shape and generated coordinates are valid"),
        };
    }

    pub(crate) fn logical_values(&self) -> Vec<T> {
        self.values().collect()
    }

    pub(crate) fn edit_values<R>(&mut self, edit: impl FnOnce(&mut [T]) -> R) -> R {
        let mut values = self.logical_values();
        let result = edit(&mut values);
        self.replace_values(values);
        result
    }

    pub(crate) fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    pub(crate) fn tensor_mut(&mut self) -> &mut Tensor<T> {
        &mut self.tensor
    }

    fn from_tensor(tensor: Tensor<T>) -> Result<Self, VectorListError> {
        if tensor.rank() != 2 {
            return Err(TensorError::ExpectedRank {
                operation: "vector-list construction",
                expected: 2,
                actual: tensor.rank(),
            }
            .into());
        }
        Ok(Self { tensor })
    }
}

impl<T> DynVectorList for VectorList<T>
where
    T: Scalar + Serialize + Copy + 'static,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn dim(&self) -> usize {
        self.dim()
    }

    fn num_vectors(&self) -> usize {
        self.num_vectors()
    }

    fn type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    fn scalar_kind(&self) -> &'static str {
        crate::math::io::json::scalar_kind::<T>()
    }

    fn clone_box(&self) -> Box<dyn DynVectorList> {
        Box::new(self.clone())
    }
}

impl<T> Serialize for VectorList<T>
where
    T: Scalar + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.tensor.serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for VectorList<T>
where
    T: Scalar + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let tensor = Tensor::<T>::deserialize(deserializer)?;
        Self::from_tensor(tensor).map_err(serde::de::Error::custom)
    }
}

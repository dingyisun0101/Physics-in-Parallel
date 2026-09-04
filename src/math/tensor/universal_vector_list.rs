//! Backend-agnostic fixed-width vector-list API.

use core::fmt;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::math::scalar::{Scalar, ScalarCastError};

use super::rank_2::vector_list::generic::DynVectorList;
use super::rank_n::errors::TensorError;
use super::universal::{Backend, Tensor, TensorBuilder, Values};

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

/// A list of fixed-width vectors with either the dense or sparse backend.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorList<T: Scalar> {
    tensor: Tensor<T>,
}

/// Safe construction buffer returned by [`VectorList::empty`].
pub struct VectorListBuilder<T: Scalar> {
    builder: TensorBuilder<T>,
}

impl<T: Scalar> VectorListBuilder<T> {
    /// Returns the selected backend.
    pub const fn backend(&self) -> Backend {
        self.builder.backend()
    }

    /// Returns `[number of vectors, dimensions]`.
    pub fn shape(&self) -> [usize; 2] {
        [self.builder.shape()[0], self.builder.shape()[1]]
    }

    /// Writes one component into the construction buffer.
    pub fn set(
        &mut self,
        vector: usize,
        component: usize,
        value: T,
    ) -> Result<(), VectorListError> {
        Ok(self.builder.set(&[vector, component], value)?)
    }

    /// Writes every component of one vector into the construction buffer.
    pub fn set_vector(&mut self, vector: usize, values: &[T]) -> Result<(), VectorListError> {
        let dimensions = self.builder.shape()[1];
        if values.len() != dimensions {
            return Err(VectorListError::VectorLength {
                expected: dimensions,
                actual: values.len(),
            });
        }
        for (component, &value) in values.iter().enumerate() {
            self.set(vector, component, value)?;
        }
        Ok(())
    }

    /// Finalizes the vector list after validating backend-specific initialization.
    pub fn finish(self) -> Result<VectorList<T>, VectorListError> {
        VectorList::from_tensor(self.builder.finish()?)
    }
}

impl<T> Eq for VectorList<T> where T: Scalar + Eq {}

impl<T: Scalar> VectorList<T> {
    /// Allocates an empty construction buffer for the selected backend.
    pub fn empty(
        dimensions: usize,
        num_vectors: usize,
        backend: Backend,
    ) -> Result<VectorListBuilder<T>, VectorListError> {
        Ok(VectorListBuilder {
            builder: Tensor::empty(&[num_vectors, dimensions], backend)?,
        })
    }

    /// Constructs vectors from row-major logical values.
    pub fn from_values(
        dimensions: usize,
        num_vectors: usize,
        backend: Backend,
        values: Vec<T>,
    ) -> Result<Self, VectorListError> {
        Self::from_tensor(Tensor::from_values(
            &[num_vectors, dimensions],
            backend,
            values,
        )?)
    }

    /// Constructs vectors from `(vector, component, value)` entries.
    pub fn from_entries(
        dimensions: usize,
        num_vectors: usize,
        backend: Backend,
        entries: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> Result<Self, VectorListError> {
        let entries = entries
            .into_iter()
            .map(|(vector, component, value)| (vec![vector, component], value));
        Self::from_tensor(Tensor::from_entries(
            &[num_vectors, dimensions],
            backend,
            entries,
        )?)
    }

    /// Constructs all-zero vectors using the selected backend.
    pub fn zeros(
        dimensions: usize,
        num_vectors: usize,
        backend: Backend,
    ) -> Result<Self, VectorListError> {
        Self::from_tensor(Tensor::zeros(&[num_vectors, dimensions], backend)?)
    }

    /// Constructs vectors filled with one value using the selected backend.
    pub fn filled(
        dimensions: usize,
        num_vectors: usize,
        backend: Backend,
        value: T,
    ) -> Result<Self, VectorListError> {
        Self::from_tensor(Tensor::filled(&[num_vectors, dimensions], backend, value)?)
    }

    /// Constructs vectors from a coordinate function using the selected backend.
    pub fn from_fn<F>(
        dimensions: usize,
        num_vectors: usize,
        backend: Backend,
        mut function: F,
    ) -> Result<Self, VectorListError>
    where
        F: FnMut(usize, usize) -> T,
    {
        Self::from_tensor(Tensor::from_fn(
            &[num_vectors, dimensions],
            backend,
            |coordinates| function(coordinates[0], coordinates[1]),
        )?)
    }

    /// Returns the selected numerical backend.
    pub const fn backend(&self) -> Backend {
        self.tensor.backend()
    }

    /// Converts this vector list to the selected backend when necessary.
    pub fn set_backend(&mut self, backend: Backend) {
        self.tensor.set_backend(backend);
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

    /// Copies one vector into caller storage without allocating.
    /// O(dimensions) time for either backend. Bounds and output length are
    /// checked before mutation; the backend is unchanged.
    pub fn vector_into(&self, vector: usize, output: &mut [T]) -> Result<(), VectorListError> {
        if vector >= self.num_vectors() {
            return Err(TensorError::CoordinateOutOfBounds {
                axis: 0,
                coordinate: vector,
                extent: self.num_vectors(),
            }
            .into());
        }
        if output.len() != self.dim() {
            return Err(VectorListError::VectorLength {
                expected: self.dim(),
                actual: output.len(),
            });
        }
        for (component, value) in output.iter_mut().enumerate() {
            *value = self
                .tensor
                .get_flat_unchecked(vector * self.dim() + component);
        }
        Ok(())
    }

    /// Copies one component across all vectors into caller storage.
    /// O(number of vectors) time, no allocation, and no writes on a length or
    /// coordinate error. Both backends have identical coordinate semantics.
    pub fn axis_into(&self, component: usize, output: &mut [T]) -> Result<(), VectorListError> {
        if component >= self.dim() {
            return Err(TensorError::CoordinateOutOfBounds {
                axis: 1,
                coordinate: component,
                extent: self.dim(),
            }
            .into());
        }
        if output.len() != self.num_vectors() {
            return Err(VectorListError::AxisLength {
                expected: self.num_vectors(),
                actual: output.len(),
            });
        }
        for (vector, value) in output.iter_mut().enumerate() {
            *value = self
                .tensor
                .get_flat_unchecked(vector * self.dim() + component);
        }
        Ok(())
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
    /// This is `O(num_vectors * dimensions)` for both backends.
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
        let dim = self.dim();
        if let Some(values) = self.tensor.dense_values_mut() {
            crate::threading::for_each_chunk_mut(values, dim, |start, chunk| {
                for (index, row) in chunk.chunks_exact_mut(dim).enumerate() {
                    T::scale_slice(row, scales[start / dim + index]);
                }
            });
        } else if scales.iter().all(|&scale| T::zero() * scale == T::zero()) {
            let entries: Vec<_> = self
                .tensor
                .sparse_entries()
                .expect("sparse backend")
                .collect();
            for (index, value) in entries {
                self.tensor
                    .set_flat_unchecked(index, value * scales[index / dim]);
            }
        } else {
            for (index, &scale) in scales.iter().enumerate() {
                for component in 0..dim {
                    let flat = index * dim + component;
                    self.tensor
                        .set_flat_unchecked(flat, self.tensor.get_flat_unchecked(flat) * scale);
                }
            }
        }
        Ok(())
    }

    /// Returns each vector's norm in the original scalar category.
    /// Uses O(number of vectors) output memory and no full input copy.
    pub fn norms(&self) -> Vec<T> {
        self.row_reductions(Scalar::norm_sqr)
            .into_iter()
            .map(Scalar::sqrt)
            .collect()
    }

    /// Returns real-valued norms, allocating only the output vector.
    pub fn norms_real(&self) -> Vec<T::Real> {
        self.row_reductions(Scalar::norm_sqr_real)
            .into_iter()
            .map(Scalar::sqrt)
            .collect()
    }

    fn row_reductions<U: Scalar>(&self, function: impl Fn(T) -> U) -> Vec<U> {
        if let Some(values) = self.tensor.dense_values() {
            return values
                .chunks_exact(self.dim())
                .map(|row| {
                    row.iter()
                        .copied()
                        .map(&function)
                        .fold(U::zero(), |a, b| a + b)
                })
                .collect();
        }
        let mut output = vec![U::zero(); self.num_vectors()];
        let mut entries: Vec<_> = self
            .tensor
            .sparse_entries()
            .expect("sparse backend")
            .collect();
        entries.sort_unstable_by_key(|&(index, _)| index);
        for (index, value) in entries {
            output[index / self.dim()] = output[index / self.dim()] + function(value);
        }
        output
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

    /// Adds two vector lists and preserves the receiver's backend.
    ///
    /// # Result backend
    ///
    /// The result has the same backend as `self`.
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

    /// Writes elementwise addition to an existing vector list.
    /// Preserves the output backend and checks shapes before mutation. Dense
    /// output reuses storage; sparse arithmetic may allocate stored entries.
    /// See [`Tensor::add_into`] for sparse cost and arithmetic semantics.
    pub fn add_into(&self, rhs: &Self, output: &mut Self) -> Result<(), VectorListError> {
        Ok(self.tensor.add_into(&rhs.tensor, &mut output.tensor)?)
    }

    /// Writes elementwise subtraction to an existing vector list.
    /// Preserves the output backend and checks shapes before mutation. Dense
    /// output reuses storage; sparse arithmetic may allocate stored entries.
    /// See [`Tensor::subtract_into`] for sparse cost and arithmetic semantics.
    pub fn subtract_into(&self, rhs: &Self, output: &mut Self) -> Result<(), VectorListError> {
        Ok(self.tensor.subtract_into(&rhs.tensor, &mut output.tensor)?)
    }

    /// Writes elementwise multiplication to an existing vector list.
    /// Preserves the output backend and checks shapes before mutation. Dense
    /// output reuses storage; sparse arithmetic may allocate stored entries.
    /// See [`Tensor::multiply_into`] for sparse cost and arithmetic semantics.
    pub fn multiply_into(&self, rhs: &Self, output: &mut Self) -> Result<(), VectorListError> {
        Ok(self.tensor.multiply_into(&rhs.tensor, &mut output.tensor)?)
    }

    /// Writes elementwise division to an existing vector list.
    /// Preserves the output backend and checks shapes before mutation. Dense
    /// output reuses storage; sparse arithmetic may allocate stored entries.
    /// See [`Tensor::divide_into`] for sparse cost and arithmetic semantics.
    pub fn divide_into(&self, rhs: &Self, output: &mut Self) -> Result<(), VectorListError> {
        Ok(self.tensor.divide_into(&rhs.tensor, &mut output.tensor)?)
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
        assert_eq!(values.len(), self.tensor.size());
        self.tensor.replace_with_values(values);
    }

    pub(crate) fn logical_values(&self) -> Vec<T> {
        self.values().collect()
    }

    pub(crate) fn edit_values<R>(&mut self, edit: impl FnOnce(&mut [T]) -> R) -> R {
        if let Some(values) = self.tensor.dense_values_mut() {
            return edit(values);
        }
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

//! Backend-agnostic rank-two matrix API.

use core::cmp::Ordering;
use core::fmt;

use num_traits::Zero;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::math::scalar::{Scalar, ScalarCastError};

use super::rank_n::errors::TensorError;
use super::universal::{Backend, Tensor, TensorBuilder, Values};

/// Failure while constructing or operating on a matrix.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MatrixError {
    Tensor(TensorError),
    InputLength { expected: usize, actual: usize },
    BatchInputLength { columns: usize, actual: usize },
    OutputLength { expected: usize, actual: usize },
}

impl From<TensorError> for MatrixError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl fmt::Display for MatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(error) => write!(formatter, "invalid matrix: {error}"),
            Self::InputLength { expected, actual } => write!(
                formatter,
                "matrix input length mismatch: expected {expected}, got {actual}"
            ),
            Self::BatchInputLength { columns, actual } => write!(
                formatter,
                "batched matrix input length {actual} is not divisible by {columns} columns"
            ),
            Self::OutputLength { expected, actual } => write!(
                formatter,
                "matrix output length mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for MatrixError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

/// A rank-two matrix with either the dense or sparse backend.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T: Scalar> {
    tensor: Tensor<T>,
}

/// Safe construction buffer returned by [`Matrix::empty`].
pub struct MatrixBuilder<T: Scalar> {
    builder: TensorBuilder<T>,
}

impl<T: Scalar> MatrixBuilder<T> {
    /// Returns the selected backend.
    pub const fn backend(&self) -> Backend {
        self.builder.backend()
    }

    /// Returns `[rows, columns]`.
    pub fn shape(&self) -> [usize; 2] {
        [self.builder.shape()[0], self.builder.shape()[1]]
    }

    /// Writes one matrix element into the construction buffer.
    pub fn set(&mut self, row: usize, column: usize, value: T) -> Result<(), MatrixError> {
        Ok(self.builder.set(&[row, column], value)?)
    }

    /// Finalizes the matrix after validating backend-specific initialization.
    pub fn finish(self) -> Result<Matrix<T>, MatrixError> {
        Matrix::from_tensor(self.builder.finish()?)
    }
}

impl<T> Eq for Matrix<T> where T: Scalar + Eq {}

impl<T: Scalar> Matrix<T> {
    /// Allocates an empty construction buffer for the selected backend.
    pub fn empty(
        rows: usize,
        columns: usize,
        backend: Backend,
    ) -> Result<MatrixBuilder<T>, MatrixError> {
        Ok(MatrixBuilder {
            builder: Tensor::empty(&[rows, columns], backend)?,
        })
    }

    /// Constructs a matrix from row-major logical values.
    pub fn from_values(
        rows: usize,
        columns: usize,
        backend: Backend,
        values: Vec<T>,
    ) -> Result<Self, MatrixError> {
        Self::from_tensor(Tensor::from_values(&[rows, columns], backend, values)?)
    }

    /// Constructs a matrix from strict `(row, column, value)` entries.
    pub fn from_entries(
        rows: usize,
        columns: usize,
        backend: Backend,
        entries: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> Result<Self, MatrixError> {
        let entries = entries
            .into_iter()
            .map(|(row, column, value)| (vec![row, column], value));
        Self::from_tensor(Tensor::from_entries(&[rows, columns], backend, entries)?)
    }

    /// Constructs an all-zero matrix using the selected backend.
    pub fn zeros(rows: usize, columns: usize, backend: Backend) -> Result<Self, MatrixError> {
        Self::from_tensor(Tensor::zeros(&[rows, columns], backend)?)
    }

    /// Constructs a matrix filled with one value using the selected backend.
    pub fn filled(
        rows: usize,
        columns: usize,
        backend: Backend,
        value: T,
    ) -> Result<Self, MatrixError> {
        Self::from_tensor(Tensor::filled(&[rows, columns], backend, value)?)
    }

    /// Constructs a matrix from a coordinate function using the selected backend.
    pub fn from_fn<F>(
        rows: usize,
        columns: usize,
        backend: Backend,
        mut function: F,
    ) -> Result<Self, MatrixError>
    where
        F: FnMut(usize, usize) -> T,
    {
        Self::from_tensor(Tensor::from_fn(&[rows, columns], backend, |coordinates| {
            function(coordinates[0], coordinates[1])
        })?)
    }

    /// Constructs an identity matrix using the selected backend.
    pub fn identity(size: usize, backend: Backend) -> Result<Self, MatrixError> {
        Self::from_tensor(Tensor::identity(size, backend)?)
    }

    /// Returns the selected numerical backend.
    pub const fn backend(&self) -> Backend {
        self.tensor.backend()
    }

    /// Converts this matrix to the selected backend when necessary.
    pub fn set_backend(&mut self, backend: Backend) {
        self.tensor.set_backend(backend);
    }

    pub fn rows(&self) -> usize {
        self.tensor.shape()[0]
    }

    pub fn columns(&self) -> usize {
        self.tensor.shape()[1]
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows(), self.columns()]
    }

    pub fn size(&self) -> usize {
        self.tensor.size()
    }

    pub fn get(&self, row: usize, column: usize) -> Result<T, MatrixError> {
        Ok(self.tensor.get(&[row, column])?)
    }

    pub fn set(&mut self, row: usize, column: usize, value: T) -> Result<(), MatrixError> {
        Ok(self.tensor.set(&[row, column], value)?)
    }

    /// Traverses every logical value in row-major coordinate order.
    ///
    /// # Complexity
    ///
    /// This is `O(rows * columns)` for both backends.
    pub fn values(&self) -> Values<'_, T> {
        self.tensor.values()
    }

    pub fn fill(&mut self, value: T) {
        self.tensor.fill(value);
    }

    /// Adds two matrices and preserves the receiver's backend.
    ///
    /// # Result backend
    ///
    /// The result has the same backend as `self`.
    pub fn add(&self, rhs: &Self) -> Result<Self, MatrixError> {
        Self::from_tensor(self.tensor.add(&rhs.tensor)?)
    }

    pub fn subtract(&self, rhs: &Self) -> Result<Self, MatrixError> {
        Self::from_tensor(self.tensor.subtract(&rhs.tensor)?)
    }

    pub fn multiply(&self, rhs: &Self) -> Result<Self, MatrixError> {
        Self::from_tensor(self.tensor.multiply(&rhs.tensor)?)
    }

    pub fn divide(&self, rhs: &Self) -> Result<Self, MatrixError> {
        Self::from_tensor(self.tensor.divide(&rhs.tensor)?)
    }

    pub fn add_into(&self, rhs: &Self, output: &mut Self) -> Result<(), MatrixError> {
        Ok(self.tensor.add_into(&rhs.tensor, &mut output.tensor)?)
    }

    pub fn subtract_into(&self, rhs: &Self, output: &mut Self) -> Result<(), MatrixError> {
        Ok(self.tensor.subtract_into(&rhs.tensor, &mut output.tensor)?)
    }

    pub fn multiply_into(&self, rhs: &Self, output: &mut Self) -> Result<(), MatrixError> {
        Ok(self.tensor.multiply_into(&rhs.tensor, &mut output.tensor)?)
    }

    pub fn divide_into(&self, rhs: &Self, output: &mut Self) -> Result<(), MatrixError> {
        Ok(self.tensor.divide_into(&rhs.tensor, &mut output.tensor)?)
    }

    pub fn scale(&self, scalar: T) -> Self {
        Self {
            tensor: self.tensor.scale(scalar),
        }
    }

    pub fn abs(&self) -> Self {
        Self {
            tensor: self.tensor.abs(),
        }
    }

    pub fn transpose(&self) -> Result<Self, MatrixError> {
        Self::from_tensor(self.tensor.transpose()?)
    }

    pub fn hermitian_transpose(&self) -> Result<Self, MatrixError> {
        Self::from_tensor(self.tensor.hermitian_transpose()?)
    }

    /// Multiplies two matrices and preserves the receiver's backend.
    ///
    /// # Result backend
    ///
    /// The result has the same backend as `self`.
    ///
    /// # Complexity
    ///
    /// The general kernel is `O(mkn)`. A sparse receiver can produce
    /// dense occupancy and dense-scale memory use.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, MatrixError> {
        Self::from_tensor(self.tensor.matmul(&rhs.tensor)?)
    }

    /// Multiplies into reusable output with the caller's selected backend.
    /// See [`Tensor::matmul_into`] for complexity and error guarantees.
    pub fn matmul_into(&self, rhs: &Self, output: &mut Self) -> Result<(), MatrixError> {
        Ok(self.tensor.matmul_into(&rhs.tensor, &mut output.tensor)?)
    }

    pub fn trace(&self) -> T {
        (0..self.rows().min(self.columns())).fold(T::zero(), |sum, coordinate| {
            sum + self
                .tensor
                .get_flat_unchecked(coordinate * self.columns() + coordinate)
        })
    }

    pub fn max_abs_real(&self) -> T::Real
    where
        T::Real: PartialOrd,
    {
        self.values()
            .map(Scalar::abs_real)
            .fold(T::Real::zero(), |left, right| {
                match left.partial_cmp(&right) {
                    Some(Ordering::Less) => right,
                    Some(_) => left,
                    None => left + right,
                }
            })
    }

    pub fn mul_vector_into(&self, input: &[T], output: &mut [T]) -> Result<(), MatrixError> {
        if input.len() != self.columns() {
            return Err(MatrixError::InputLength {
                expected: self.columns(),
                actual: input.len(),
            });
        }
        if output.len() != self.rows() {
            return Err(MatrixError::OutputLength {
                expected: self.rows(),
                actual: output.len(),
            });
        }
        self.mul_vector_unchecked(input, output);
        Ok(())
    }

    fn mul_vector_unchecked(&self, input: &[T], output: &mut [T]) {
        if let Some(values) = self.tensor.dense_values() {
            crate::threading::for_each_chunk_mut(output, 1, |start, chunk| {
                for (row, out) in chunk.iter_mut().enumerate() {
                    *out = values[(start + row) * self.columns()..][..self.columns()]
                        .iter()
                        .zip(input)
                        .fold(T::zero(), |sum, (&a, &b)| sum + a * b);
                }
            });
        } else {
            for (row, out) in output.iter_mut().enumerate() {
                *out = input
                    .iter()
                    .copied()
                    .enumerate()
                    .fold(T::zero(), |sum, (column, value)| {
                        sum + self
                            .tensor
                            .get_flat_unchecked(row * self.columns() + column)
                            * value
                    });
            }
        }
    }

    /// Applies a matrix to consecutive input vectors, reusing caller output.
    /// O(batch * rows * columns) time and no dense scratch allocation.
    /// All lengths are validated before the first output write.
    pub fn mul_vectors_into(&self, input: &[T], output: &mut [T]) -> Result<(), MatrixError> {
        if !input.len().is_multiple_of(self.columns()) {
            return Err(MatrixError::BatchInputLength {
                columns: self.columns(),
                actual: input.len(),
            });
        }
        let batch = input.len() / self.columns();
        let expected = batch
            .checked_mul(self.rows())
            .ok_or(TensorError::ShapeProductOverflow {
                shape: vec![batch, self.rows()],
            })?;
        if output.len() != expected {
            return Err(MatrixError::OutputLength {
                expected,
                actual: output.len(),
            });
        }
        for (input, output) in input
            .chunks_exact(self.columns())
            .zip(output.chunks_exact_mut(self.rows()))
        {
            self.mul_vector_unchecked(input, output);
        }
        Ok(())
    }

    pub fn cast<U: Scalar>(&self) -> Result<Matrix<U>, ScalarCastError> {
        Ok(Matrix {
            tensor: self.tensor.cast()?,
        })
    }

    pub(crate) fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    pub(crate) fn tensor_mut(&mut self) -> &mut Tensor<T> {
        &mut self.tensor
    }

    fn from_tensor(tensor: Tensor<T>) -> Result<Self, MatrixError> {
        if tensor.rank() != 2 {
            return Err(TensorError::ExpectedRank {
                operation: "matrix construction",
                expected: 2,
                actual: tensor.rank(),
            }
            .into());
        }
        Ok(Self { tensor })
    }
}

impl<T> Serialize for Matrix<T>
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

impl<'de, T> Deserialize<'de> for Matrix<T>
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

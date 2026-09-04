/*!
Generic user-facing matrix wrapper.

`Matrix<T, B>` is a rank-2 mathematical facade over a backend `B`. The backend
may be ordinary rank-N dense/sparse tensor storage or a structured matrix
backend that stores only canonical entries and derives the rest from symmetry.
*/

use core::cmp::Ordering;
use core::fmt;
use core::marker::PhantomData;

use crate::math::scalar::{Scalar, ScalarCastError};
use crate::math::tensor::rank_n::{Dense, Sparse, Tensor};
use crate::threading::{parallel_chunk_len, should_parallelize_operations};
use num_traits::Zero;
use rayon::prelude::*;

use super::matrix_backend_trait::{MatrixBackend, wrap_axis_index};

#[inline]
fn max_or_propagate_unordered<R>(left: R, right: R) -> R
where
    R: Scalar<Real = R> + PartialOrd,
{
    match left.partial_cmp(&right) {
        Some(Ordering::Less) => right,
        Some(_) => left,
        None => left + right,
    }
}

/// Rank-N dense tensor storage used as a matrix backend.
#[derive(Debug, Clone)]
pub struct RankNDense<T: Scalar> {
    tensor: Tensor<T, Dense>,
}

/// Rank-N sparse tensor storage used as a matrix backend.
#[derive(Debug, Clone)]
pub struct RankNSparse<T: Scalar> {
    tensor: Tensor<T, Sparse>,
}

/// Generic matrix facade.
#[derive(Debug, Clone)]
pub struct Matrix<T: Scalar, B: MatrixBackend<T> = RankNDense<T>> {
    backend: B,
    _scalar: PhantomData<T>,
}

/// Dense rank-N-backed matrix.
pub type DenseMatrix<T> = Matrix<T, RankNDense<T>>;

/// Sparse rank-N-backed matrix.
/// Failure while constructing a matrix or applying it to a vector.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MatrixError {
    /// Matrix dimensions must both be nonzero.
    InvalidShape { rows: usize, cols: usize },
    /// The logical element count cannot be represented by `usize`.
    ShapeProductOverflow { rows: usize, cols: usize },
    /// Matrix axes or logical size cannot be represented by signed accessors.
    IndexSpaceOverflow { rows: usize, cols: usize },
    /// Row-major storage did not contain exactly `rows * cols` elements.
    DataLengthMismatch { expected: usize, actual: usize },
    /// The input vector length did not match the matrix column count.
    InputLength { expected: usize, actual: usize },
    /// A batched input did not contain a whole number of column-sized vectors.
    BatchInputLength { columns: usize, actual: usize },
    /// The output vector length did not match the matrix row count.
    OutputLength { expected: usize, actual: usize },
}

impl fmt::Display for MatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape { rows, cols } => {
                write!(formatter, "matrix shape must be nonzero, got {rows}x{cols}")
            }
            Self::ShapeProductOverflow { rows, cols } => {
                write!(
                    formatter,
                    "matrix shape {rows}x{cols} overflows its element count"
                )
            }
            Self::IndexSpaceOverflow { rows, cols } => write!(
                formatter,
                "matrix shape {rows}x{cols} exceeds the signed index space"
            ),
            Self::DataLengthMismatch { expected, actual } => write!(
                formatter,
                "matrix data length mismatch: expected {expected}, got {actual}"
            ),
            Self::InputLength { expected, actual } => write!(
                formatter,
                "matrix input length mismatch: expected {expected}, got {actual}"
            ),
            Self::BatchInputLength { columns, actual } => write!(
                formatter,
                "batched matrix input length {actual} is not divisible by the column count {columns}"
            ),
            Self::OutputLength { expected, actual } => write!(
                formatter,
                "matrix output length mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for MatrixError {}

impl<T: Scalar, B: MatrixBackend<T>> Matrix<T, B> {
    /// Wrap an already-constructed backend in the public matrix facade.
    #[inline]
    pub(crate) fn from_backend(backend: B) -> Self {
        Self {
            backend,
            _scalar: PhantomData,
        }
    }

    /// Borrows the complete logical matrix buffer when the selected backend
    /// exposes contiguous row-major storage.
    #[inline]
    pub(crate) fn contiguous_data(&self) -> Option<&[T]> {
        self.backend.contiguous_data()
    }

    #[inline]
    pub(crate) fn contiguous_data_mut(&mut self) -> Option<&mut [T]> {
        self.backend.contiguous_data_mut()
    }

    /// Borrows the backend for format-specific crate-internal IO.
    #[inline]
    pub(crate) fn backend(&self) -> &B {
        &self.backend
    }

    /// Construct a zero matrix with backend-specific storage.
    #[inline]
    pub fn empty(rows: usize, cols: usize) -> Self {
        Self::from_backend(B::empty(rows, cols))
    }

    /// Alias for `empty`.
    #[inline]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::empty(rows, cols)
    }

    /// Number of logical rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.backend.rows()
    }

    /// Number of logical columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.backend.cols()
    }

    /// Largest scalar magnitude across all logical matrix entries.
    ///
    /// Dense and sparse backends reduce their native storage in parallel.
    /// Structured backends reduce the complete logical matrix so implicit
    /// entries introduced by symmetry are included. An unordered magnitude
    /// (for example, a floating-point NaN) propagates to the result.
    pub fn max_abs_real(&self) -> T::Real
    where
        T::Real: PartialOrd,
    {
        if let Some(values) = self.backend.contiguous_data() {
            return values
                .par_iter()
                .with_min_len(parallel_chunk_len(values.len()).unwrap_or(1))
                .map(|value| value.abs_real())
                .reduce(T::Real::zero, max_or_propagate_unordered);
        }
        if let Some(entries) = self.backend.sparse_entries() {
            return entries
                .into_par_iter()
                .with_min_len(parallel_chunk_len(self.size()).unwrap_or(1))
                .map(|(_, value)| value.abs_real())
                .reduce(T::Real::zero, max_or_propagate_unordered);
        }

        let cols = self.cols();
        (0..self.size())
            .into_par_iter()
            .with_min_len(parallel_chunk_len(self.size()).unwrap_or(1))
            .map(|index| {
                self.get((index / cols) as isize, (index % cols) as isize)
                    .abs_real()
            })
            .reduce(T::Real::zero, max_or_propagate_unordered)
    }

    /// Logical matrix shape as `[rows, cols]`.
    #[inline]
    pub fn shape(&self) -> [usize; 2] {
        self.backend.shape()
    }

    /// Number of logical entries.
    #[inline]
    pub fn size(&self) -> usize {
        self.backend.size()
    }

    /// Read the scalar at `(row, col)`.
    #[inline]
    pub fn get(&self, row: isize, col: isize) -> T {
        self.backend.get(row, col)
    }

    /// Read through a signed row-major index with periodic wrapping.
    #[inline]
    pub fn get_flat(&self, flat: isize) -> T {
        let flat = wrap_axis_index(flat, self.size());
        self.get((flat / self.cols()) as isize, (flat % self.cols()) as isize)
    }

    /// Computes `output = self * input` without allocating.
    ///
    /// Dense backends read their contiguous row-major storage directly. Other
    /// backends use logical scalar access, preserving their storage semantics.
    /// Diagonal storage takes O(n) work for finite inputs; nonfinite input
    /// retains logical zero-product semantics through the general O(n²) path.
    pub fn mul_vector_into(&self, input: &[T], output: &mut [T]) -> Result<(), MatrixError> {
        if input.len() != self.cols() {
            return Err(MatrixError::InputLength {
                expected: self.cols(),
                actual: input.len(),
            });
        }
        if output.len() != self.rows() {
            return Err(MatrixError::OutputLength {
                expected: self.rows(),
                actual: output.len(),
            });
        }

        if let Some(diagonal) = self.backend.diagonal_data()
            && input.iter().all(|value| value.is_finite())
        {
            for ((out, &coefficient), &value) in output.iter_mut().zip(diagonal).zip(input) {
                *out = T::zero() + coefficient * value;
            }
            return Ok(());
        }

        if let Some(data) = self.contiguous_data() {
            for (row, output_value) in data.chunks_exact(self.cols()).zip(output.iter_mut()) {
                *output_value = row
                    .iter()
                    .copied()
                    .zip(input.iter().copied())
                    .map(|(coefficient, value)| coefficient * value)
                    .sum();
            }
        } else {
            for (row, output_value) in output.iter_mut().enumerate() {
                *output_value = input
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(column, value)| self.get(row as isize, column as isize) * value)
                    .sum();
            }
        }
        Ok(())
    }

    /// Applies this matrix to a contiguous batch of vectors without allocating.
    ///
    /// `input` is interpreted as consecutive vectors of length `cols()`. The
    /// batch size is inferred, and `output` must contain the same number of
    /// consecutive vectors of length `rows()`.
    pub fn mul_vectors_into(&self, input: &[T], output: &mut [T]) -> Result<(), MatrixError> {
        if !input.len().is_multiple_of(self.cols()) {
            return Err(MatrixError::BatchInputLength {
                columns: self.cols(),
                actual: input.len(),
            });
        }
        let batch = input.len() / self.cols();
        let expected_output =
            batch
                .checked_mul(self.rows())
                .ok_or(MatrixError::ShapeProductOverflow {
                    rows: batch,
                    cols: self.rows(),
                })?;
        if output.len() != expected_output {
            return Err(MatrixError::OutputLength {
                expected: expected_output,
                actual: output.len(),
            });
        }
        if batch == 0 {
            return Ok(());
        }

        let rows = self.rows();
        let cols = self.cols();
        let operations = batch.saturating_mul(rows).saturating_mul(cols);
        if batch < 2 || !should_parallelize_operations(operations) {
            if let Some(data) = self.contiguous_data() {
                for (output_vector, input_vector) in
                    output.chunks_exact_mut(rows).zip(input.chunks_exact(cols))
                {
                    for (matrix_row, output_value) in
                        data.chunks_exact(cols).zip(output_vector.iter_mut())
                    {
                        *output_value = matrix_row
                            .iter()
                            .copied()
                            .zip(input_vector.iter().copied())
                            .map(|(coefficient, value)| coefficient * value)
                            .sum();
                    }
                }
            } else {
                for (output_vector, input_vector) in
                    output.chunks_exact_mut(rows).zip(input.chunks_exact(cols))
                {
                    for (row, output_value) in output_vector.iter_mut().enumerate() {
                        *output_value = input_vector
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(column, value)| self.get(row as isize, column as isize) * value)
                            .sum();
                    }
                }
            }
            return Ok(());
        }
        let min_vectors_per_job = parallel_chunk_len(batch).unwrap_or(1);
        if let Some(data) = self.contiguous_data() {
            output
                .par_chunks_mut(rows)
                .zip(input.par_chunks(cols))
                .with_min_len(min_vectors_per_job)
                .for_each(|(output_vector, input_vector)| {
                    for (matrix_row, output_value) in
                        data.chunks_exact(cols).zip(output_vector.iter_mut())
                    {
                        *output_value = matrix_row
                            .iter()
                            .copied()
                            .zip(input_vector.iter().copied())
                            .map(|(coefficient, value)| coefficient * value)
                            .sum();
                    }
                });
        } else {
            output
                .par_chunks_mut(rows)
                .zip(input.par_chunks(cols))
                .with_min_len(min_vectors_per_job)
                .for_each(|(output_vector, input_vector)| {
                    for (row, output_value) in output_vector.iter_mut().enumerate() {
                        *output_value = input_vector
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(column, value)| self.get(row as isize, column as isize) * value)
                            .sum();
                    }
                });
        }
        Ok(())
    }

    /// Write the scalar at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: isize, col: isize, value: T) {
        self.backend.set(row, col, value);
    }

    /// Write through a signed row-major index with periodic wrapping.
    #[inline]
    pub fn set_flat(&mut self, flat: isize, value: T) {
        let flat = wrap_axis_index(flat, self.size());
        self.set(
            (flat / self.cols()) as isize,
            (flat % self.cols()) as isize,
            value,
        );
    }

    /// Fill every logical entry according to backend semantics.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.backend.fill(value);
    }

    fn to_dense_rank_n_tensor(&self) -> Tensor<T, Dense>
    where
        T: Copy,
    {
        Tensor::<T, Dense>::from_fn(&[self.rows(), self.cols()], |idx| self.get(idx[0], idx[1]))
    }

    fn dense_matrix_from_rank_n(tensor: Tensor<T, Dense>) -> DenseMatrix<T>
    where
        T: Copy,
    {
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        DenseMatrix::from_vec(rows, cols, tensor.data().to_vec())
    }

    fn matrix_from_dense_rank_n_preserving(tensor: &Tensor<T, Dense>) -> Self
    where
        T: Copy,
    {
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        let entries: Vec<(usize, usize, T)> = (0..rows * cols)
            .into_par_iter()
            .with_min_len(parallel_chunk_len(rows * cols).unwrap_or(1))
            .map(|k| {
                let row = k / cols;
                let col = k % cols;
                (row, col, tensor.get(&[row as isize, col as isize]))
            })
            .collect();

        let mut out = Self::empty(rows, cols);
        for (row, col, value) in entries {
            out.set(row as isize, col as isize, value);
        }
        out
    }

    fn zip_preserving<RhsBackend, F>(&self, rhs: &Matrix<T, RhsBackend>, f: F) -> Self
    where
        RhsBackend: MatrixBackend<T>,
        T: Copy + Send + Sync,
        F: Fn(T, T) -> T + Sync + Send,
    {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "matrix elementwise operation shape mismatch"
        );

        let rows = self.rows();
        let cols = self.cols();
        let mut out = Self::empty(rows, cols);
        if let Some(values) = out.contiguous_data_mut() {
            let min_elements_per_job = parallel_chunk_len(values.len()).unwrap_or(1);
            values
                .par_iter_mut()
                .with_min_len(min_elements_per_job)
                .enumerate()
                .for_each(|(flat, value)| {
                    *value = f(self.get_flat(flat as isize), rhs.get_flat(flat as isize));
                });
            return out;
        }

        let entries: Vec<(usize, usize, T)> = (0..rows * cols)
            .into_par_iter()
            .with_min_len(parallel_chunk_len(rows * cols).unwrap_or(1))
            .map(|k| {
                let row = k / cols;
                let col = k % cols;
                let value = f(
                    self.get(row as isize, col as isize),
                    rhs.get(row as isize, col as isize),
                );
                (row, col, value)
            })
            .collect();
        for (row, col, value) in entries {
            out.set(row as isize, col as isize, value);
        }
        out
    }

    /// Materialize this matrix as an ordinary dense rank-N-backed matrix.
    pub fn to_dense_matrix(&self) -> DenseMatrix<T>
    where
        T: Copy,
    {
        Self::dense_matrix_from_rank_n(self.to_dense_rank_n_tensor())
    }

    /// Type-preserving elementwise matrix addition.
    ///
    /// The returned matrix has the same scalar type and same backend as
    /// `self`. Structured backends keep enforcing their mathematical
    /// invariants, so this method panics if the computed result cannot be
    /// represented by the left-hand backend.
    pub fn add<RhsBackend>(&self, rhs: &Matrix<T, RhsBackend>) -> Self
    where
        RhsBackend: MatrixBackend<T>,
        T: Copy + Send + Sync,
    {
        self.zip_preserving(rhs, |a, b| a + b)
    }

    /// Type-preserving elementwise matrix subtraction.
    pub fn sub<RhsBackend>(&self, rhs: &Matrix<T, RhsBackend>) -> Self
    where
        RhsBackend: MatrixBackend<T>,
        T: Copy + Send + Sync,
    {
        self.zip_preserving(rhs, |a, b| a - b)
    }

    /// Type-preserving elementwise matrix multiplication.
    pub fn elem_mul<RhsBackend>(&self, rhs: &Matrix<T, RhsBackend>) -> Self
    where
        RhsBackend: MatrixBackend<T>,
        T: Copy + Send + Sync,
    {
        self.zip_preserving(rhs, |a, b| a * b)
    }

    /// Type-preserving elementwise matrix division.
    pub fn elem_div<RhsBackend>(&self, rhs: &Matrix<T, RhsBackend>) -> Self
    where
        RhsBackend: MatrixBackend<T>,
        T: Copy + Send + Sync,
    {
        self.zip_preserving(rhs, |a, b| a / b)
    }

    /// Type-preserving scalar multiplication.
    pub fn scalar_mul(&self, scalar: T) -> Self
    where
        T: Copy + Send + Sync,
    {
        let rows = self.rows();
        let cols = self.cols();
        let mut out = Self::empty(rows, cols);
        if let Some(values) = out.contiguous_data_mut() {
            let min_elements_per_job = parallel_chunk_len(values.len()).unwrap_or(1);
            values
                .par_iter_mut()
                .with_min_len(min_elements_per_job)
                .enumerate()
                .for_each(|(flat, value)| {
                    *value = self.get_flat(flat as isize) * scalar;
                });
            return out;
        }

        let entries: Vec<(usize, usize, T)> = (0..rows * cols)
            .into_par_iter()
            .with_min_len(parallel_chunk_len(rows * cols).unwrap_or(1))
            .map(|k| {
                let row = k / cols;
                let col = k % cols;
                (row, col, self.get(row as isize, col as isize) * scalar)
            })
            .collect();
        for (row, col, value) in entries {
            out.set(row as isize, col as isize, value);
        }
        out
    }

    /// Type-preserving transpose.
    ///
    /// This method keeps the same backend type as `self`; structured backends
    /// therefore reject transposes that no longer fit their declared support.
    /// Use `transpose_to_dense` when changing backend is acceptable.
    pub fn transpose(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        let transposed = self.to_dense_rank_n_tensor().transpose();
        Self::matrix_from_dense_rank_n_preserving(&transposed)
    }

    /// Type-preserving Hermitian transpose.
    ///
    /// The result keeps the same backend type as `self`. Use
    /// `hermitian_transpose_to_dense` for the explicit backend-converting form.
    pub fn hermitian_transpose(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        let transposed = self.to_dense_rank_n_tensor().hermitian_transpose();
        Self::matrix_from_dense_rank_n_preserving(&transposed)
    }

    /// Type-preserving matrix multiplication.
    ///
    /// The calculation uses the rank-N matrix multiplication algorithm, then
    /// writes the result back into the left-hand backend. Structured backends
    /// panic if the product cannot be represented by that backend.
    pub fn matmul<RhsBackend>(&self, rhs: &Matrix<T, RhsBackend>) -> Self
    where
        RhsBackend: MatrixBackend<T>,
        T: Copy + Send + Sync,
    {
        assert_eq!(
            self.cols(),
            rhs.rows(),
            "matrix multiplication dimension mismatch"
        );
        let lhs = self.to_dense_rank_n_tensor();
        let rhs = rhs.to_dense_rank_n_tensor();
        let product = lhs.matmul(&rhs);
        Self::matrix_from_dense_rank_n_preserving(&product)
    }

    /// Backend-converting elementwise matrix addition.
    pub fn add_to_dense<RhsBackend>(&self, rhs: &Matrix<T, RhsBackend>) -> DenseMatrix<T>
    where
        RhsBackend: MatrixBackend<T>,
        T: Copy + Send + Sync,
    {
        let lhs = self.to_dense_rank_n_tensor();
        let rhs = rhs.to_dense_rank_n_tensor();
        Self::dense_matrix_from_rank_n(lhs.zip_with(&rhs, |a, b| a + b))
    }

    /// Backend-converting elementwise matrix subtraction.
    pub fn sub_to_dense<RhsBackend>(&self, rhs: &Matrix<T, RhsBackend>) -> DenseMatrix<T>
    where
        RhsBackend: MatrixBackend<T>,
        T: Copy + Send + Sync,
    {
        let lhs = self.to_dense_rank_n_tensor();
        let rhs = rhs.to_dense_rank_n_tensor();
        Self::dense_matrix_from_rank_n(lhs.zip_with(&rhs, |a, b| a - b))
    }

    /// Backend-converting elementwise matrix multiplication.
    pub fn elem_mul_to_dense<RhsBackend>(&self, rhs: &Matrix<T, RhsBackend>) -> DenseMatrix<T>
    where
        RhsBackend: MatrixBackend<T>,
        T: Copy + Send + Sync,
    {
        let lhs = self.to_dense_rank_n_tensor();
        let rhs = rhs.to_dense_rank_n_tensor();
        Self::dense_matrix_from_rank_n(lhs.elem_mul(&rhs))
    }

    /// Backend-converting elementwise matrix division.
    pub fn elem_div_to_dense<RhsBackend>(&self, rhs: &Matrix<T, RhsBackend>) -> DenseMatrix<T>
    where
        RhsBackend: MatrixBackend<T>,
        T: Copy + Send + Sync,
    {
        let lhs = self.to_dense_rank_n_tensor();
        let rhs = rhs.to_dense_rank_n_tensor();
        Self::dense_matrix_from_rank_n(lhs.elem_div(&rhs))
    }

    /// Backend-converting scalar multiplication.
    pub fn scalar_mul_to_dense(&self, scalar: T) -> DenseMatrix<T>
    where
        T: Copy + Send + Sync,
    {
        Self::dense_matrix_from_rank_n(self.to_dense_rank_n_tensor().scalar_mul(scalar))
    }

    /// Backend-converting transpose.
    pub fn transpose_to_dense(&self) -> DenseMatrix<T>
    where
        T: Copy + Send + Sync,
    {
        Self::dense_matrix_from_rank_n(self.to_dense_rank_n_tensor().transpose())
    }

    /// Backend-converting Hermitian transpose.
    pub fn hermitian_transpose_to_dense(&self) -> DenseMatrix<T>
    where
        T: Copy + Send + Sync,
    {
        Self::dense_matrix_from_rank_n(self.to_dense_rank_n_tensor().hermitian_transpose())
    }

    /// Backend-converting matrix multiplication.
    pub fn matmul_to_dense<RhsBackend>(&self, rhs: &Matrix<T, RhsBackend>) -> DenseMatrix<T>
    where
        RhsBackend: MatrixBackend<T>,
        T: Copy + Send + Sync,
    {
        assert_eq!(
            self.cols(),
            rhs.rows(),
            "matrix multiplication dimension mismatch"
        );
        let lhs = self.to_dense_rank_n_tensor();
        let rhs = rhs.to_dense_rank_n_tensor();
        Self::dense_matrix_from_rank_n(lhs.matmul(&rhs))
    }

    /// Sum of diagonal entries.
    pub fn trace(&self) -> T
    where
        T: Copy,
    {
        let n = self.rows().min(self.cols());
        (0..n).fold(T::zero(), |acc, i| acc + self.get(i as isize, i as isize))
    }
}

impl<T: Scalar> Matrix<T, RankNDense<T>> {
    /// Return the elementwise absolute value using the dense tensor backend.
    #[inline]
    pub fn abs(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        Self::from_backend(RankNDense {
            tensor: self.backend.tensor.abs(),
        })
    }

    /// Builds a dense matrix from checked row-major storage.
    pub fn try_from_vec(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::InvalidShape { rows, cols });
        }
        let expected = rows
            .checked_mul(cols)
            .ok_or(MatrixError::ShapeProductOverflow { rows, cols })?;
        if rows > isize::MAX as usize
            || cols > isize::MAX as usize
            || expected > isize::MAX as usize
        {
            return Err(MatrixError::IndexSpaceOverflow { rows, cols });
        }
        if data.len() != expected {
            return Err(MatrixError::DataLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self::from_backend(RankNDense {
            tensor: Tensor::<T, Dense>::from_vec(&[rows, cols], data),
        }))
    }

    /// Build a dense matrix from row-major values.
    #[inline]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        Self::try_from_vec(rows, cols, data)
            .unwrap_or_else(|error| panic!("DenseMatrix::from_vec: {error}"))
    }

    /// Build a dense matrix from a coordinate function.
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let data = (0..rows)
            .flat_map(|row| (0..cols).map(move |col| (row, col)))
            .map(|(row, col)| f(row, col))
            .collect();
        Self::from_vec(rows, cols, data)
    }

    /// Fallibly cast all entries to a new scalar type.
    #[inline]
    pub fn try_cast_to<U: Scalar>(&self) -> Result<Matrix<U, RankNDense<U>>, ScalarCastError> {
        self.backend
            .tensor
            .try_cast_to::<U>()
            .map(|tensor| Matrix::from_backend(RankNDense { tensor }))
    }

    /// Cast all entries to a new scalar type, panicking on conversion failure.
    #[inline]
    pub fn cast_to<U: Scalar + Send + Sync>(&self) -> Matrix<U, RankNDense<U>> {
        Matrix::from_backend(RankNDense {
            tensor: self.backend.tensor.cast_to::<U>(),
        })
    }

    /// Convert this dense matrix into a sparse rank-N-backed matrix.
    #[inline]
    pub fn to_sparse(&self) -> Matrix<T, RankNSparse<T>> {
        Matrix::from_backend(RankNSparse {
            tensor: self.backend.tensor.to_sparse(),
        })
    }
}

impl<T: Scalar> Matrix<T, RankNSparse<T>> {
    /// Build a sparse matrix from `(row, col, value)` entries.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> Self {
        let tensor_triplets = triplets
            .into_iter()
            .map(|(row, col, value)| (vec![row, col], value));
        Self::from_backend(RankNSparse {
            tensor: Tensor::<T, Sparse>::from_triplets(vec![rows, cols], tensor_triplets),
        })
    }

    /// Fallibly cast all explicitly stored entries to a new scalar type.
    #[inline]
    pub fn try_cast_to<U: Scalar>(&self) -> Result<Matrix<U, RankNSparse<U>>, ScalarCastError> {
        self.backend
            .tensor
            .try_cast_to::<U>()
            .map(|tensor| Matrix::from_backend(RankNSparse { tensor }))
    }

    /// Cast all explicitly stored entries to a new scalar type, panicking on failure.
    #[inline]
    pub fn cast_to<U: Scalar + Send + Sync>(&self) -> Matrix<U, RankNSparse<U>> {
        Matrix::from_backend(RankNSparse {
            tensor: self.backend.tensor.cast_to::<U>(),
        })
    }

    /// Convert this sparse matrix into a dense rank-N-backed matrix.
    #[inline]
    pub fn to_dense(&self) -> Matrix<T, RankNDense<T>> {
        Matrix::from_backend(RankNDense {
            tensor: self.backend.tensor.to_dense(),
        })
    }

    /// Number of explicitly stored sparse entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.backend.tensor.nnz()
    }
}

impl<T: Scalar> MatrixBackend<T> for RankNDense<T> {
    #[inline]
    fn empty(rows: usize, cols: usize) -> Self {
        Self {
            tensor: Tensor::<T, Dense>::empty(&[rows, cols]),
        }
    }

    #[inline]
    fn rows(&self) -> usize {
        self.tensor.shape()[0]
    }

    #[inline]
    fn cols(&self) -> usize {
        self.tensor.shape()[1]
    }

    #[inline]
    fn get(&self, row: isize, col: isize) -> T {
        self.tensor.get(&[row, col])
    }

    #[inline]
    fn set(&mut self, row: isize, col: isize, value: T) {
        self.tensor.set(&[row, col], value);
    }

    #[inline]
    fn contiguous_data(&self) -> Option<&[T]> {
        Some(self.tensor.data())
    }

    #[inline]
    fn contiguous_data_mut(&mut self) -> Option<&mut [T]> {
        Some(self.tensor.storage_mut().data_mut())
    }
}

impl<T: Scalar> MatrixBackend<T> for RankNSparse<T> {
    #[inline]
    fn empty(rows: usize, cols: usize) -> Self {
        Self {
            tensor: Tensor::<T, Sparse>::empty(&[rows, cols]),
        }
    }

    #[inline]
    fn rows(&self) -> usize {
        self.tensor.shape()[0]
    }

    #[inline]
    fn cols(&self) -> usize {
        self.tensor.shape()[1]
    }

    #[inline]
    fn get(&self, row: isize, col: isize) -> T {
        self.tensor.get(&[row, col])
    }

    #[inline]
    fn set(&mut self, row: isize, col: isize, value: T) {
        self.tensor.set(&[row, col], value);
    }

    #[inline]
    fn sparse_entries(&self) -> Option<Vec<(usize, T)>> {
        Some(
            self.tensor
                .storage()
                .iter()
                .map(|(&index, &value)| (index, value))
                .collect(),
        )
    }
}

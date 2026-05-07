/*!
Generic user-facing matrix wrapper.

`Matrix<T, B>` is a rank-2 mathematical facade over a backend `B`. The backend
may be ordinary rank-N dense/sparse tensor storage or a structured matrix
backend that stores only canonical entries and derives the rest from symmetry.
*/

use core::marker::PhantomData;

use crate::math::scalar::{Scalar, ScalarCastError};
use crate::math::tensor::rank_n::{Dense, Sparse, Tensor};
use rayon::prelude::*;

use super::matrix_backend_trait::MatrixBackend;

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
pub type SparseMatrix<T> = Matrix<T, RankNSparse<T>>;

impl<T: Scalar, B: MatrixBackend<T>> Matrix<T, B> {
    /// Wrap an already-constructed backend in the public matrix facade.
    #[inline]
    pub(crate) fn from_backend(backend: B) -> Self {
        Self {
            backend,
            _scalar: PhantomData,
        }
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

    /// Write the scalar at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: isize, col: isize, value: T) {
        self.backend.set(row, col, value);
    }

    /// Fill every logical entry according to backend semantics.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.backend.fill(value);
    }

    /// Print a compact matrix table.
    pub fn print(&self)
    where
        T: Copy,
    {
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                print!("{}\t", self.get(row as isize, col as isize));
            }
            println!();
        }
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
        DenseMatrix::from_fn(rows, cols, |row, col| {
            tensor.get(&[row as isize, col as isize])
        })
    }

    fn matrix_from_dense_rank_n_preserving(tensor: &Tensor<T, Dense>) -> Self
    where
        T: Copy,
    {
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        let entries: Vec<(usize, usize, T)> = (0..rows * cols)
            .into_par_iter()
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
        let entries: Vec<(usize, usize, T)> = (0..rows * cols)
            .into_par_iter()
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

        let mut out = Self::empty(rows, cols);
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
        let entries: Vec<(usize, usize, T)> = (0..rows * cols)
            .into_par_iter()
            .map(|k| {
                let row = k / cols;
                let col = k % cols;
                (row, col, self.get(row as isize, col as isize) * scalar)
            })
            .collect();

        let mut out = Self::empty(rows, cols);
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
    /// Build a dense matrix from row-major values.
    #[inline]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            rows.checked_mul(cols)
                .expect("matrix shape product overflow"),
            "dense matrix data length mismatch"
        );
        Self::from_backend(RankNDense {
            tensor: Tensor::<T, Dense>::from_vec(&[rows, cols], data),
        })
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
}

/*!
Backend contract for rank-2 matrix storage.

`MatrixBackend<T>` is intentionally smaller than the public matrix API. It
describes the minimum behavior needed by the generic `Matrix<T, B>` wrapper:
shape, scalar access, and scalar update. Dense, sparse, and structured matrix
representations can therefore share one user-facing wrapper while keeping their
own storage invariants.
*/

use crate::math::scalar::Scalar;

/// Storage behavior required by the generic matrix wrapper.
pub trait MatrixBackend<T: Scalar>: Send + Sync + Clone {
    /// Construct an all-zero matrix with backend-specific storage.
    fn empty(rows: usize, cols: usize) -> Self;

    /// Number of logical rows.
    fn rows(&self) -> usize;

    /// Number of logical columns.
    fn cols(&self) -> usize;

    /// Read the scalar at `(row, col)`.
    ///
    /// Implementations should use the same periodic index semantics as the
    /// rank-N tensor layer unless a future backend explicitly documents a
    /// stricter indexing policy.
    fn get(&self, row: isize, col: isize) -> T;

    /// Write the scalar at `(row, col)`.
    ///
    /// Structured backends may reject writes that violate their mathematical
    /// invariants, such as a nonzero value outside a triangular support.
    fn set(&mut self, row: isize, col: isize, value: T);

    /// Borrows a complete row-major buffer when this backend stores one.
    ///
    /// Backends without contiguous logical storage return `None`. Consumers
    /// must then use [`MatrixBackend::get`] to visit logical entries instead
    /// of assuming a particular physical representation.
    #[inline]
    fn contiguous_data(&self) -> Option<&[T]> {
        None
    }

    /// Mutably borrows a complete row-major buffer when this backend stores one.
    #[inline]
    fn contiguous_data_mut(&mut self) -> Option<&mut [T]> {
        None
    }

    /// Returns explicitly stored row-major entries for a genuinely sparse backend.
    ///
    /// Structured matrices return `None` because their canonical storage does
    /// not describe the complete logical matrix without applying symmetry or
    /// triangular rules.
    #[inline]
    fn sparse_entries(&self) -> Option<Vec<(usize, T)>> {
        None
    }

    /// Return `[rows, cols]`.
    #[inline]
    fn shape(&self) -> [usize; 2] {
        [self.rows(), self.cols()]
    }

    /// Number of logical matrix entries.
    #[inline]
    fn size(&self) -> usize {
        self.rows()
            .checked_mul(self.cols())
            .expect("matrix shape product overflow")
    }

    /// Fill every logical entry according to backend semantics.
    ///
    /// Generic dense and sparse rank-N backends update through their native
    /// tensor fill behavior. Structured backends preserve their invariants.
    fn fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                self.set(row as isize, col as isize, value);
            }
        }
    }
}

/// Periodically wrap a signed row or column index into `[0, dim)`.
#[inline(always)]
pub(crate) fn wrap_axis_index(idx: isize, dim: usize) -> usize {
    debug_assert!(dim > 0);
    let dim_signed = dim as isize;
    let mut wrapped = idx % dim_signed;
    if wrapped < 0 {
        wrapped += dim_signed;
    }
    wrapped as usize
}

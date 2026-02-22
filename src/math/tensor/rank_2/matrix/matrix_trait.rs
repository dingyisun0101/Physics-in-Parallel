// src/math/tensor/rank_2/matrix/matrix_trait.rs
/*!
    Common **2D matrix interface** for dense/sparse backends.

    **Keep row/col view types internal to each backend.**
        This trait only exposes **zero-copy row accessors** as plain slices for maximum
        performance on dense, row-major storage. Columns (typically strided) and any
        sparse-specific views should be implemented **inside** each backend module.

    ### Notes
        - `row/col_view()` / `row/col_view_mut()` must return slices that directly borrow the
        backend’s internal storage (no copies).
        - Implementations should document index normalization (e.g., whether negative
        indices are supported/wrapped).
*/

use crate::math::{
    scalar::Scalar, 
};


// ============================================================================
// -------------------------------- Trait Def ---------------------------------
// ============================================================================
pub trait MatrixTrait<T: Scalar>: Send + Sync + Clone {
    // ----------------------------------- Shape -------------------------------
    fn rows(&self) -> usize;

    fn cols(&self) -> usize;

    #[inline]
    fn shape(&self) -> [usize; 2] { [self.rows(), self.cols()] }

    // ----------------------------- Element access ----------------------------
    /// Get **by value** at `(i, j)`. Backends may choose to wrap/normalize indices.
    fn get(&self, i: isize, j: isize) -> T;

    /// Set the value at `(i, j)`.
    fn set(&mut self, i: isize, j: isize, val: T);

    // ------------------------------ Zero-copy views --------------------------
    /// Immutable **row** view as a contiguous slice; length = `cols()`.
    fn row_view<'a>(&'a self, i: isize) -> &'a [T];

    /// Mutable **row** view as a contiguous slice; length = `cols()`.
    fn row_view_mut<'a>(&'a mut self, i: isize) -> &'a mut [T];

    /// Immutable **column** view as a contiguous slice; length = `rows()`.
    fn col_view<'a>(&'a self, j: isize) -> &'a [T];

    /// Mutable **column** view as a contiguous slice; length = `rows()`.
    fn col_view_mut<'a>(&'a mut self, j: isize) -> &'a mut [T];

    // ------------------------------- Bulk helpers ---------------------------
    /// Set entire row `i` from `vals` (length must equal `cols()`).
    fn set_row_from_slice(&mut self, i: isize, vals: &[T]);

    /// Set entire column `j` from `vals` (length must equal `rows()`).
    fn set_col_from_slice(&mut self, j: isize, vals: &[T]);

    // ---------------------------------- I/O --------------------------------
    fn print(&self);

    fn to_string(&self);

    // ------------------------------- Basic Linalg ------------------------------
    fn transpose(&mut self);

    fn clamp(&mut self, min_val: T, max_val: T) where T: PartialOrd;

    fn normalize(&mut self);

    fn normalize_by_max(&mut self) where T: PartialOrd;
}

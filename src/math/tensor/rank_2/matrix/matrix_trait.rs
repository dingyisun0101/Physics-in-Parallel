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
    /// Annotation:
    /// - Purpose: Executes `rows` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn rows(&self) -> usize;

    /// Annotation:
    /// - Purpose: Executes `cols` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn cols(&self) -> usize;

    #[inline]
    /// Annotation:
    /// - Purpose: Returns the logical shape metadata.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn shape(&self) -> [usize; 2] { [self.rows(), self.cols()] }

    // ----------------------------- Element access ----------------------------
    /// Get **by value** at `(i, j)`. Backends may choose to wrap/normalize indices.
    /// Annotation:
    /// - Purpose: Executes `get` logic for this module.
    /// - Parameters:
    ///   - `i` (`isize`): Primary index argument.
    ///   - `j` (`isize`): Secondary index argument.
    fn get(&self, i: isize, j: isize) -> T;

    /// Set the value at `(i, j)`.
    /// Annotation:
    /// - Purpose: Executes `set` logic for this module.
    /// - Parameters:
    ///   - `i` (`isize`): Primary index argument.
    ///   - `j` (`isize`): Secondary index argument.
    ///   - `val` (`T`): Value provided by caller for write/update behavior.
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
    /// Annotation:
    /// - Purpose: Sets the `row_from_slice` value.
    /// - Parameters:
    ///   - `i` (`isize`): Primary index argument.
    ///   - `vals` (`&[T]`): Parameter of type `&[T]` used by `set_row_from_slice`.
    fn set_row_from_slice(&mut self, i: isize, vals: &[T]);

    /// Set entire column `j` from `vals` (length must equal `rows()`).
    /// Annotation:
    /// - Purpose: Sets the `col_from_slice` value.
    /// - Parameters:
    ///   - `j` (`isize`): Secondary index argument.
    ///   - `vals` (`&[T]`): Parameter of type `&[T]` used by `set_col_from_slice`.
    fn set_col_from_slice(&mut self, j: isize, vals: &[T]);

    // ---------------------------------- I/O --------------------------------
    /// Annotation:
    /// - Purpose: Prints a human-readable representation.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn print(&self);

    /// Annotation:
    /// - Purpose: Converts this value into `string` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn to_string(&self);

    // ------------------------------- Basic Linalg ------------------------------
    /// Annotation:
    /// - Purpose: Executes `transpose` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn transpose(&mut self);

    /// Annotation:
    /// - Purpose: Executes `clamp` logic for this module.
    /// - Parameters:
    ///   - `min_val` (`T`): Value provided by caller for write/update behavior.
    ///   - `max_val` (`T`): Value provided by caller for write/update behavior.
    fn clamp(&mut self, min_val: T, max_val: T) where T: PartialOrd;

    /// Annotation:
    /// - Purpose: Executes `normalize` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn normalize(&mut self);

    /// Annotation:
    /// - Purpose: Executes `normalize_by_max` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn normalize_by_max(&mut self) where T: PartialOrd;
}

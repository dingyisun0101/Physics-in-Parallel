
use std::path::PathBuf;
use crate::math::prelude::ScalarSerde;


// ======================================================================================
/*                                   Space Trait                                       */
// ======================================================================================
/*
Only the signatures relevant to this grid are shown.
Coordinates are `isize`; negatives are handled by **wrap** (periodic) or **clamp** (non-periodic).

- `dims()` returns `[d, l]` (rank and per-axis length). The total site count is `l^d`.
- `linear_size()` equals `cfg.num_sites()`.
*/
pub trait Space<T: ScalarSerde> {
    /// Borrow the backing slice.
    /// Annotation:
    /// - Purpose: Executes `data` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn data(&self) -> &[T];
    /// Return `[d, l]`. (Total sites is **not** stored here; it’s `l^d`.)
    /// Annotation:
    /// - Purpose: Executes `dims` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn dims(&self) -> Vec<usize>;
    /// `l^d`: total number of sites in the grid.
    /// Annotation:
    /// - Purpose: Executes `linear_size` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn linear_size(&self) -> usize;

    /// Safe read at multi-index `coord`.
    /// Annotation:
    /// - Purpose: Executes `get` logic for this module.
    /// - Parameters:
    ///   - `coord` (`&[isize]`): Coordinate input used for spatial addressing.
    fn get(&self, coord: &[isize]) -> &T;
    /// Safe mutable read at multi-index `coord`.
    /// Annotation:
    /// - Purpose: Returns the `mut` value.
    /// - Parameters:
    ///   - `coord` (`&[isize]`): Coordinate input used for spatial addressing.
    fn get_mut(&mut self, coord: &[isize]) -> &mut T;
    /// Safe write at multi-index `coord`.
    /// Annotation:
    /// - Purpose: Executes `set` logic for this module.
    /// - Parameters:
    ///   - `coord` (`&[isize]`): Coordinate input used for spatial addressing.
    ///   - `val` (`T`): Value provided by caller for write/update behavior.
    fn set(&mut self, coord: &[isize], val: T);

    /// Save the grid after **optional downscaling** to side length `l_target`.
    ///
    /// The `output_dir` is treated as a **file path** here (not a directory).
    /// Annotation:
    /// - Purpose: Executes `save` logic for this module.
    /// - Parameters:
    ///   - `output_file` (`&PathBuf`): Parameter of type `&PathBuf` used by `save`.
    ///   - `l_target` (`usize`): Parameter of type `usize` used by `save`.
    fn save(&self, output_file: &PathBuf, l_target: usize) -> std::io::Result<()>;

    /// Fill the entire grid with a single value (parallel).
    /// Annotation:
    /// - Purpose: Sets the `all` value.
    /// - Parameters:
    ///   - `val` (`T`): Value provided by caller for write/update behavior.
    fn set_all(&mut self, val: T);
}



use std::path::PathBuf;
use crate::math::scalar::Scalar;


// ======================================================================================
/*                                   Space Trait                                       */
// ======================================================================================
/*
Only the signatures relevant to this grid are shown.
Coordinates are `isize`; negatives are handled by **wrap** (periodic) or **clamp** (non-periodic).

- `dims()` returns `[d, l]` (rank and per-axis length). The total site count is `l^d`.
- `linear_size()` equals `cfg.num_sites()`.
*/
pub trait Space<T: Scalar> {
    /// Borrow the backing slice.
    fn data(&self) -> &[T];
    /// Return `[d, l]`. (Total sites is **not** stored here; it’s `l^d`.)
    fn dims(&self) -> Vec<usize>;
    /// `l^d`: total number of sites in the grid.
    fn linear_size(&self) -> usize;

    /// Safe read at multi-index `coord`.
    fn get(&self, coord: &[isize]) -> &T;
    /// Safe mutable read at multi-index `coord`.
    fn get_mut(&mut self, coord: &[isize]) -> &mut T;
    /// Safe write at multi-index `coord`.
    fn set(&mut self, coord: &[isize], val: T);

    /// Save the grid after **optional downscaling** to side length `l_target`.
    ///
    /// The `output_dir` is treated as a **file path** here (not a directory).
    fn save(&self, output_file: &PathBuf, l_target: usize) -> std::io::Result<()>;

    /// Fill the entire grid with a single value (parallel).
    fn set_all(&mut self, val: T);
}



/*!
Shared interface for physical spaces.

Purpose:
    This trait gives higher-level simulation code one consistent way to read,
    update, fill, and save a spatial container without knowing the container's
    concrete geometry. A discrete cubic lattice, a future triangular lattice,
    and a future continuous-space discretization can all expose the same basic
    operations while keeping their own boundary rules and storage choices
    private.

Coordinate contract:
    Coordinates are passed as `isize` slices so callers can naturally express
    physically meaningful out-of-domain locations such as `-1` or `l`. The
    concrete space decides how those coordinates are interpreted. For example,
    a lattice with periodic boundary conditions wraps them, while a reflective
    lattice mirrors them at the wall.

Data contract:
    The returned slice is a read-only view of the space's compact backing data.
    It is useful for inspection, serialization, or aggregate calculations. It
    does not promise a universal geometric layout across all future space
    types; coordinate methods remain the stable way to address physical sites.
*/

use std::path::PathBuf;

use crate::math::prelude::ScalarSerde;

pub trait Space<T: ScalarSerde> {
    /// Borrow the compact backing data for inspection or serialization.
    fn data(&self) -> &[T];

    /// Return the shape metadata needed to understand this space.
    ///
    /// For a square lattice this is the tensor-style shape, such as `[128]`,
    /// `[64, 64]`, or `[32, 64, 16]`.
    fn dims(&self) -> Vec<usize>;

    /// Return the number of physical sites represented by this space.
    fn linear_size(&self) -> usize;

    /// Borrow the value at a physical coordinate after this space applies its
    /// boundary condition or coordinate normalization rule.
    fn get(&self, coord: &[isize]) -> &T;

    /// Mutably borrow the value at a physical coordinate after this space
    /// applies its boundary condition or coordinate normalization rule.
    fn get_mut(&mut self, coord: &[isize]) -> &mut T;

    /// Store a value at a physical coordinate after this space applies its
    /// boundary condition or coordinate normalization rule.
    fn set(&mut self, coord: &[isize], val: T);

    /// Save this space to an external file, optionally reducing the side length
    /// for formats that support lattice-style downsampling.
    fn save(&self, output_file: &PathBuf, l_target: usize) -> std::io::Result<()>;

    /// Fill every represented site with the same value.
    fn set_all(&mut self, val: T);
}

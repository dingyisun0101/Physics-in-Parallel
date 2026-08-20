//! Reusable row-major tensor layout and coordinate conversion.

use super::errors::{TensorResult, checked_num_elements, ensure_index_rank};

/// Converts a signed coordinate to a row-major flat index with periodic
/// normalization, without constructing a cached layout.
pub fn flat_index_wrapped(shape: &[usize], coordinate: &[isize]) -> TensorResult<usize> {
    ensure_index_rank(shape, coordinate.len())?;
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (&extent, &component) in shape.iter().rev().zip(coordinate.iter().rev()) {
        flat += wrap_axis_index(component, extent) * stride;
        stride *= extent;
    }
    Ok(flat)
}

/// Periodically normalizes a signed row-major position over a linear domain.
#[inline]
pub fn normalize_flat_index(index: isize, size: usize) -> usize {
    debug_assert!(size > 0 && size <= isize::MAX as usize);
    index.rem_euclid(size as isize) as usize
}

/// Validated row-major shape metadata shared by tensor-facing algorithms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RowMajorLayout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    size: usize,
}

impl RowMajorLayout {
    /// Validates a shape and precomputes its row-major strides.
    pub fn try_new(shape: &[usize]) -> TensorResult<Self> {
        let size = checked_num_elements(shape)?;
        let mut strides = vec![1; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            size,
        })
    }

    /// Convenience constructor for already trusted tensor shapes.
    pub fn new(shape: &[usize]) -> Self {
        Self::try_new(shape).expect("invalid row-major tensor shape")
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub const fn size(&self) -> usize {
        self.size
    }

    /// Converts a signed coordinate to a row-major flat index with periodic
    /// normalization on each axis.
    pub fn flat_index_wrapped(&self, coordinate: &[isize]) -> TensorResult<usize> {
        ensure_index_rank(&self.shape, coordinate.len())?;
        Ok(self
            .shape
            .iter()
            .zip(&self.strides)
            .zip(coordinate)
            .map(|((&extent, &stride), &component)| wrap_axis_index(component, extent) * stride)
            .sum())
    }

    /// Panicking convenience for tensor backends whose rank is already known.
    #[inline]
    pub fn index_wrapped(&self, coordinate: &[isize]) -> usize {
        self.flat_index_wrapped(coordinate)
            .expect("tensor index rank mismatch")
    }

    /// Periodically normalizes a signed index over the complete linear domain.
    #[inline]
    pub fn flat_index(&self, index: isize) -> usize {
        normalize_flat_index(index, self.size)
    }

    /// Converts one valid flat index into an owned signed coordinate.
    pub fn coordinate(&self, flat: usize) -> Option<Vec<isize>> {
        if flat >= self.size {
            return None;
        }
        let mut coordinate = vec![0; self.shape.len()];
        self.coordinate_into(flat, &mut coordinate);
        Some(coordinate)
    }

    /// Writes one valid flat index into caller-owned coordinate storage.
    ///
    /// Panics when `flat` is outside the layout or the output rank differs.
    #[inline]
    pub fn coordinate_into(&self, flat: usize, coordinate: &mut [isize]) {
        assert!(flat < self.size, "flat tensor index out of bounds");
        assert_eq!(
            coordinate.len(),
            self.shape.len(),
            "tensor coordinate rank mismatch"
        );
        let mut remainder = flat;
        for axis in (1..self.shape.len()).rev() {
            let extent = self.shape[axis];
            coordinate[axis] = (remainder % extent) as isize;
            remainder /= extent;
        }
        coordinate[0] = remainder as isize;
    }
}

#[inline]
fn wrap_axis_index(index: isize, extent: usize) -> usize {
    debug_assert!(extent > 0);
    index.rem_euclid(extent as isize) as usize
}

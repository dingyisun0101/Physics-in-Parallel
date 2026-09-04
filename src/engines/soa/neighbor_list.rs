/*!
Cell-linked candidate-pair generator for structure-of-arrays particle data.

Purpose:
`NeighborList` divides a finite rectangular domain into cells and stores object
indices in those cells. It then emits unique unordered candidate pairs from the
same or adjacent cells. Candidate pairs are useful because they greatly reduce
the number of pair distances a particle model must check.

Important semantics:
This module does not apply a physical cutoff distance. It only says that two
objects live in nearby cells and therefore may be close enough for a downstream
model to inspect. Particle-level wrappers should perform the actual distance
test and apply domain-specific filters such as alive masks.

Positions outside the configured bounds are clipped to the nearest valid cell.
The list is non-periodic; periodic or reflective boundary policies belong in
space or model-level code.
*/

use ahash::AHashMap;
use core::fmt;
use std::ops::Range;

/// Errors returned by neighbor-list construction and rebuild operations.
#[derive(Debug, Clone, PartialEq)]
pub enum NeighborListError {
    /// Derived geometry or position size cannot be represented.
    Capacity { context: &'static str },
    /// Nonfinite position rejected before changing existing buckets.
    NonfinitePosition { index: usize },
    /// Cell width is not finite or is not strictly positive.
    InvalidCellWidth {
        /// Requested cell width.
        cell_width: f64,
    },
    /// Domain bounds are malformed on one axis.
    InvalidBounds {
        /// Axis where bounds failed validation.
        axis: usize,
        /// Lower bound value.
        min: f64,
        /// Upper bound value.
        max: f64,
    },
    /// Flat position array length does not match `dim * n_objects`.
    InvalidPositionShape {
        /// Required flat length.
        expected_len: usize,
        /// Supplied flat length.
        got_len: usize,
    },
}

impl fmt::Display for NeighborListError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity { context } => {
                write!(f, "neighbor-list {context} exceeds representable capacity")
            }
            Self::NonfinitePosition { index } => {
                write!(f, "nonfinite position at flat index {index}")
            }
            Self::InvalidCellWidth { cell_width } => write!(
                f,
                "neighbor-list cell width must be finite and positive; got {cell_width}"
            ),
            Self::InvalidBounds { axis, min, max } => write!(
                f,
                "neighbor-list bounds on axis {axis} must be finite with min < max; got min={min}, max={max}"
            ),
            Self::InvalidPositionShape {
                expected_len,
                got_len,
            } => write!(
                f,
                "flat position data has length {got_len}; expected {expected_len}"
            ),
        }
    }
}

impl std::error::Error for NeighborListError {}

/// Cell-linked list that emits candidate object pairs from neighboring cells.
#[derive(Debug, Clone)]
pub struct NeighborList {
    /// Spatial dimension inferred from `min`/`max`.
    dim: usize,
    /// Lower bound of each axis.
    min: Vec<f64>,
    /// Upper bound of each axis.
    max: Vec<f64>,
    /// Width of one cell along each axis.
    cell_width: f64,
    /// Number of cells along each axis.
    cells_per_axis: Vec<usize>,
    /// Row-major-like strides for converting cell coordinates to ids.
    strides: Vec<usize>,
    /// Offsets of every same/adjacent cell in `[-1, 0, 1]^dim`.
    neighbor_offsets: Vec<Vec<isize>>,
    /// Contiguous (cell id, object id) records, sorted by both keys.
    occupants: Vec<(usize, usize)>,
    /// Sorted occupied cell ids and ranges into occupants.
    cells: Vec<(usize, Range<usize>)>,
    /// Hash lookup into sorted cell metadata; never determines visitation order.
    cell_index: AHashMap<usize, usize>,
    logical_cells: usize,
}

impl NeighborList {
    /// Builds a nonperiodic list storing only occupied cells.
    /// Derived counts/strides are checked before allocation. High-rank grids
    /// whose stencil exceeds 65,536 offsets compare occupied cells instead;
    /// that fallback costs O(occupied_cells² × dimension) per query.
    pub fn new(min: &[f64], max: &[f64], cell_width: f64) -> Result<Self, NeighborListError> {
        if !cell_width.is_finite() || cell_width <= 0.0 {
            return Err(NeighborListError::InvalidCellWidth { cell_width });
        }
        if min.len() != max.len() || min.is_empty() {
            return Err(NeighborListError::InvalidBounds {
                axis: 0,
                min: min.first().copied().unwrap_or(0.0),
                max: max.first().copied().unwrap_or(0.0),
            });
        }

        let dim = min.len();
        let mut cells_per_axis = vec![0usize; dim];
        for axis in 0..dim {
            let lo = min[axis];
            let hi = max[axis];
            if !lo.is_finite() || !hi.is_finite() || hi <= lo {
                return Err(NeighborListError::InvalidBounds {
                    axis,
                    min: lo,
                    max: hi,
                });
            }
            let count = ((hi - lo) / cell_width).ceil();
            if !(hi - lo).is_finite() || !count.is_finite() || count >= isize::MAX as f64 {
                return Err(NeighborListError::Capacity {
                    context: "axis cell count",
                });
            }
            cells_per_axis[axis] = (count as usize).max(1);
        }

        let mut strides = Vec::with_capacity(dim);
        let mut logical_cells = 1usize;
        for &count in &cells_per_axis {
            strides.push(logical_cells);
            logical_cells =
                logical_cells
                    .checked_mul(count)
                    .ok_or(NeighborListError::Capacity {
                        context: "cell grid",
                    })?;
        }
        // High ranks use occupied-cell comparisons instead of allocating 3^dim offsets.
        let stencil_size = (0..dim).try_fold(1usize, |size, _| size.checked_mul(3));
        let neighbor_offsets = if stencil_size.is_some_and(|size| size <= 65_536) {
            build_neighbor_offsets(dim)
        } else {
            Vec::new()
        };

        Ok(Self {
            dim,
            min: min.to_vec(),
            max: max.to_vec(),
            cell_width,
            cells_per_axis,
            strides,
            neighbor_offsets,
            occupants: Vec::new(),
            cells: Vec::new(),
            cell_index: AHashMap::new(),
            logical_cells,
        })
    }

    /// Returns spatial dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns lower domain bounds.
    pub fn min(&self) -> &[f64] {
        self.min.as_slice()
    }

    /// Returns upper domain bounds.
    pub fn max(&self) -> &[f64] {
        self.max.as_slice()
    }

    /// Returns configured cell width.
    pub fn cell_width(&self) -> f64 {
        self.cell_width
    }

    /// Returns number of cells along each axis.
    pub fn cells_per_axis(&self) -> &[usize] {
        self.cells_per_axis.as_slice()
    }

    /// Returns the logical grid cell count, including unstored empty cells.
    pub fn num_cells(&self) -> usize {
        self.logical_cells
    }

    /// Returns number of objects currently stored across all cells.
    pub fn num_objects(&self) -> usize {
        self.occupants.len()
    }

    /// Removes all stored ids while retaining contiguous buffers and hash capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.occupants.clear();
        self.cells.clear();
        self.cell_index.clear();
    }

    /// Rebuilds cell buckets from flat row-major position data.
    ///
    /// Nonfinite positions and layout errors leave existing buckets unchanged.
    /// Memory scales with occupied cells and objects, not logical grid size.
    /// Records are sorted in reusable contiguous storage; no per-cell vector is
    /// allocated. Rebuild costs O(n log n) worst-case, with retained capacity.
    /// Hash lookup accelerates queries while sorted records retain pair order.
    /// `positions` must have length `dim * n_objects`, with object `i` stored at
    /// `positions[i * dim .. (i + 1) * dim]`.
    pub fn rebuild(
        &mut self,
        positions: &[f64],
        n_objects: usize,
    ) -> Result<(), NeighborListError> {
        let expected_len = self
            .dim
            .checked_mul(n_objects)
            .ok_or(NeighborListError::Capacity {
                context: "position layout",
            })?;
        if positions.len() != expected_len {
            return Err(NeighborListError::InvalidPositionShape {
                expected_len,
                got_len: positions.len(),
            });
        }

        if let Some(index) = positions.iter().position(|value| !value.is_finite()) {
            return Err(NeighborListError::NonfinitePosition { index });
        }
        self.clear();
        self.occupants.reserve(n_objects);
        for (index, row) in positions.chunks_exact(self.dim).enumerate() {
            let id = row
                .iter()
                .enumerate()
                .map(|(axis, &x)| self.coord_along_axis(x, axis) * self.strides[axis])
                .sum();
            self.occupants.push((id, index));
        }
        self.occupants.sort_unstable();
        let mut start = 0;
        while start < self.occupants.len() {
            let id = self.occupants[start].0;
            let mut end = start + 1;
            while end < self.occupants.len() && self.occupants[end].0 == id {
                end += 1;
            }
            self.cells.push((id, start..end));
            start = end;
        }
        self.cell_index.reserve(self.cells.len());
        for (index, (id, _)) in self.cells.iter().enumerate() {
            self.cell_index.insert(*id, index);
        }

        Ok(())
    }

    /// Visits unique unordered candidate pairs `(i, j)` with `i < j`.
    ///
    /// The callback receives candidates from the same or adjacent cells. The
    /// result is not distance-filtered.
    pub fn for_each_pair_candidate<F>(&self, mut f: F)
    where
        F: FnMut(usize, usize),
    {
        let mut coord = vec![0usize; self.dim];
        let mut nbr = vec![0usize; self.dim];

        for (cell_index, (cell_id, range)) in self.cells.iter().enumerate() {
            self.coord_from_linear_id(*cell_id, &mut coord);
            let objects = &self.occupants[range.clone()];
            if self.neighbor_offsets.is_empty() {
                for (other_id, other_range) in &self.cells[cell_index..] {
                    self.coord_from_linear_id(*other_id, &mut nbr);
                    if coord.iter().zip(&nbr).all(|(a, b)| a.abs_diff(*b) <= 1) {
                        emit_pairs(
                            cell_id == other_id,
                            objects,
                            &self.occupants[other_range.clone()],
                            &mut f,
                        );
                    }
                }
            } else {
                for offset in &self.neighbor_offsets {
                    if !self.try_offset_coord(&coord, offset, &mut nbr) {
                        continue;
                    }
                    let other_id = self.linear_id(&nbr);
                    if other_id < *cell_id {
                        continue;
                    }
                    if let Some(&index) = self.cell_index.get(&other_id) {
                        let other_range = self.cells[index].1.clone();
                        emit_pairs(
                            other_id == *cell_id,
                            objects,
                            &self.occupants[other_range],
                            &mut f,
                        );
                    }
                }
            }
        }
    }

    /// Collects unique unordered candidate pairs into a vector.
    pub fn collect_pair_candidates(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        self.for_each_pair_candidate(|i, j| pairs.push((i, j)));
        pairs
    }

    #[inline]
    fn coord_along_axis(&self, x: f64, axis: usize) -> usize {
        let lo = self.min[axis];
        let hi = self.max[axis];
        let span = hi - lo;
        if !x.is_finite() {
            return 0;
        }
        let clipped = x.clamp(lo, hi - f64::EPSILON.min(span * 1e-12));
        let t = ((clipped - lo) / self.cell_width).floor() as isize;
        t.clamp(0, (self.cells_per_axis[axis] as isize) - 1) as usize
    }

    #[inline]
    fn linear_id(&self, coord: &[usize]) -> usize {
        let mut id = 0usize;
        for (axis, &value) in coord.iter().enumerate().take(self.dim) {
            id += value * self.strides[axis];
        }
        id
    }

    #[inline]
    fn coord_from_linear_id(&self, mut id: usize, coord: &mut [usize]) {
        for axis in (0..self.dim).rev() {
            let stride = self.strides[axis];
            coord[axis] = id / stride;
            id %= stride;
        }
    }

    #[inline]
    fn try_offset_coord(&self, base: &[usize], off: &[isize], out: &mut [usize]) -> bool {
        for axis in 0..self.dim {
            let v = (base[axis] as isize) + off[axis];
            if v < 0 || v >= self.cells_per_axis[axis] as isize {
                return false;
            }
            out[axis] = v as usize;
        }
        true
    }
}

fn emit_pairs(
    same: bool,
    left: &[(usize, usize)],
    right: &[(usize, usize)],
    f: &mut impl FnMut(usize, usize),
) {
    for (offset, &(_, i)) in left.iter().enumerate() {
        for &(_, j) in if same { &right[offset + 1..] } else { right } {
            f(i.min(j), i.max(j));
        }
    }
}

fn build_neighbor_offsets(dim: usize) -> Vec<Vec<isize>> {
    let mut out = Vec::<Vec<isize>>::new();
    let mut cur = vec![0isize; dim];
    build_neighbor_offsets_rec(0, cur.as_mut_slice(), &mut out);
    out
}

fn build_neighbor_offsets_rec(axis: usize, cur: &mut [isize], out: &mut Vec<Vec<isize>>) {
    if axis == cur.len() {
        out.push(cur.to_vec());
        return;
    }
    for v in [-1isize, 0, 1] {
        cur[axis] = v;
        build_neighbor_offsets_rec(axis + 1, cur, out);
    }
}

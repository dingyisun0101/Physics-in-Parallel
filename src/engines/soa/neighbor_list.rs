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

use core::fmt;

/// Errors returned by neighbor-list construction and rebuild operations.
#[derive(Debug, Clone, PartialEq)]
pub enum NeighborListError {
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
    /// Object ids stored in each cell.
    buckets: Vec<Vec<usize>>,
}

impl NeighborList {
    /// Builds an empty cell-linked list over a rectangular domain.
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
            let n_cells = ((hi - lo) / cell_width).ceil() as usize;
            cells_per_axis[axis] = n_cells.max(1);
        }

        let strides = compute_strides(cells_per_axis.as_slice());
        let n_cells_total = cells_per_axis
            .iter()
            .copied()
            .fold(1usize, |acc, v| acc.saturating_mul(v));
        let buckets = vec![Vec::new(); n_cells_total];
        let neighbor_offsets = build_neighbor_offsets(dim);

        Ok(Self {
            dim,
            min: min.to_vec(),
            max: max.to_vec(),
            cell_width,
            cells_per_axis,
            strides,
            neighbor_offsets,
            buckets,
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

    /// Returns total number of cells.
    pub fn num_cells(&self) -> usize {
        self.buckets.len()
    }

    /// Returns number of objects currently stored across all cells.
    pub fn num_objects(&self) -> usize {
        self.buckets.iter().map(Vec::len).sum()
    }

    /// Removes all stored object ids while preserving bucket allocation.
    #[inline]
    pub fn clear(&mut self) {
        for bucket in self.buckets.iter_mut() {
            bucket.clear();
        }
    }

    /// Rebuilds cell buckets from flat row-major position data.
    ///
    /// `positions` must have length `dim * n_objects`, with object `i` stored at
    /// `positions[i * dim .. (i + 1) * dim]`.
    pub fn rebuild(
        &mut self,
        positions: &[f64],
        n_objects: usize,
    ) -> Result<(), NeighborListError> {
        let expected_len = self.dim.saturating_mul(n_objects);
        if positions.len() != expected_len {
            return Err(NeighborListError::InvalidPositionShape {
                expected_len,
                got_len: positions.len(),
            });
        }

        self.clear();
        let mut coord = vec![0usize; self.dim];
        for i in 0..n_objects {
            let i0 = i * self.dim;
            for axis in 0..self.dim {
                coord[axis] = self.coord_along_axis(positions[i0 + axis], axis);
            }
            let id = self.linear_id(coord.as_slice());
            self.buckets[id].push(i);
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

        for cell_id in 0..self.buckets.len() {
            self.coord_from_linear_id(cell_id, coord.as_mut_slice());
            let cell_objects = &self.buckets[cell_id];
            if cell_objects.is_empty() {
                continue;
            }

            for off in self.neighbor_offsets.iter() {
                if !self.try_offset_coord(coord.as_slice(), off.as_slice(), nbr.as_mut_slice()) {
                    continue;
                }
                let nbr_id = self.linear_id(nbr.as_slice());
                if nbr_id < cell_id {
                    continue;
                }

                let nbr_objects = &self.buckets[nbr_id];
                if nbr_objects.is_empty() {
                    continue;
                }

                if nbr_id == cell_id {
                    for a in 0..cell_objects.len() {
                        for b in (a + 1)..cell_objects.len() {
                            f(cell_objects[a], cell_objects[b]);
                        }
                    }
                } else {
                    for &i in cell_objects {
                        for &j in nbr_objects {
                            let (a, b) = if i < j { (i, j) } else { (j, i) };
                            if a != b {
                                f(a, b);
                            }
                        }
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

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1].saturating_mul(shape[axis - 1]);
    }
    strides
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

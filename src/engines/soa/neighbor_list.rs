/*!
Cell-linked neighbor-list backend for bounded particle systems.
*/

#[derive(Debug, Clone, PartialEq)]
pub enum NeighborListError {
    InvalidCellSize { cell_size: f64 },
    InvalidBounds {
        axis: usize,
        min: f64,
        max: f64,
    },
    InvalidPositionShape {
        expected_len: usize,
        got_len: usize,
    },
}

#[derive(Debug, Clone)]
pub struct NeighborList {
    dim: usize,
    min: Vec<f64>,
    max: Vec<f64>,
    cell_size: f64,
    cells_per_axis: Vec<usize>,
    strides: Vec<usize>,
    neighbor_offsets: Vec<Vec<isize>>,
    buckets: Vec<Vec<usize>>,
}

impl NeighborList {
    pub fn new(min: &[f64], max: &[f64], cell_size: f64) -> Result<Self, NeighborListError> {
        if !cell_size.is_finite() || cell_size <= 0.0 {
            return Err(NeighborListError::InvalidCellSize { cell_size });
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
            let n_cells = ((hi - lo) / cell_size).ceil() as usize;
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
            cell_size,
            cells_per_axis,
            strides,
            neighbor_offsets,
            buckets,
        })
    }

    #[inline]
    pub fn clear(&mut self) {
        for bucket in self.buckets.iter_mut() {
            bucket.clear();
        }
    }

    pub fn rebuild(&mut self, positions: &[f64], n_particles: usize) -> Result<(), NeighborListError> {
        let expected_len = self.dim.saturating_mul(n_particles);
        if positions.len() != expected_len {
            return Err(NeighborListError::InvalidPositionShape {
                expected_len,
                got_len: positions.len(),
            });
        }

        self.clear();
        let mut coord = vec![0usize; self.dim];
        for i in 0..n_particles {
            let i0 = i * self.dim;
            for axis in 0..self.dim {
                coord[axis] = self.coord_along_axis(positions[i0 + axis], axis);
            }
            let id = self.linear_id(coord.as_slice());
            self.buckets[id].push(i);
        }
        Ok(())
    }

    pub fn for_each_candidate_pair<F>(&self, mut f: F)
    where
        F: FnMut(usize, usize),
    {
        let mut coord = vec![0usize; self.dim];
        let mut nbr = vec![0usize; self.dim];

        for cell_id in 0..self.buckets.len() {
            self.coord_from_linear_id(cell_id, coord.as_mut_slice());
            let cell_particles = &self.buckets[cell_id];
            if cell_particles.is_empty() {
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

                let nbr_particles = &self.buckets[nbr_id];
                if nbr_particles.is_empty() {
                    continue;
                }

                if nbr_id == cell_id {
                    for a in 0..cell_particles.len() {
                        for b in (a + 1)..cell_particles.len() {
                            f(cell_particles[a], cell_particles[b]);
                        }
                    }
                } else {
                    for &i in cell_particles {
                        for &j in nbr_particles {
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

    #[inline]
    fn coord_along_axis(&self, x: f64, axis: usize) -> usize {
        let lo = self.min[axis];
        let hi = self.max[axis];
        let span = hi - lo;
        if !x.is_finite() {
            return 0;
        }
        let clipped = x.clamp(lo, hi - f64::EPSILON.min(span * 1e-12));
        let t = ((clipped - lo) / self.cell_size).floor() as isize;
        t.clamp(0, (self.cells_per_axis[axis] as isize) - 1) as usize
    }

    #[inline]
    fn linear_id(&self, coord: &[usize]) -> usize {
        let mut id = 0usize;
        for axis in 0..self.dim {
            id += coord[axis] * self.strides[axis];
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

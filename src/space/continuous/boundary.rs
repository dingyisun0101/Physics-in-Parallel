/*!
Boundary conditions for continuous coordinate vectors.

Purpose:
Continuous boundaries define how points in real-valued space are returned to an
axis-aligned domain after they move outside it. This module is intentionally
independent of particles, tensors, or any particular storage backend. It works
on one coordinate vector at a time, or on a flat list of coordinate vectors.

Data shape:
- A single position is `r: &mut [f64]` with length equal to the boundary
  dimension.
- A single velocity is `v: &mut [f64]` with the same length as `r`.
- A bulk position list is flat vector-list storage:
  `[r0_x, r0_y, ..., r1_x, r1_y, ...]`.

Design:
The boundary structs own only geometric box bounds. Higher-level modules decide
how to traverse their storage. The bulk methods here use Rayon over flat vector
chunks, while particle-specific rules such as alive masks and rigid masks live
in `models::particles::boundary`.
*/

use core::fmt;

use crate::threading::parallel_chunk_len;
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryError {
    InvalidBounds {
        axis: usize,
        min: f64,
        max: f64,
    },
    InvalidVectorDimension {
        label: &'static str,
        expected: usize,
        got: usize,
    },
    InvalidFlatVectorListLength {
        label: &'static str,
        dim: usize,
        len: usize,
    },
    InconsistentFlatVectorListLength {
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for BoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds { axis, min, max } => write!(
                f,
                "boundary bounds on axis {axis} must be finite with min < max; got min={min}, max={max}"
            ),
            Self::InvalidVectorDimension {
                label,
                expected,
                got,
            } => write!(f, "{label} vector has dimension {got}; expected {expected}"),
            Self::InvalidFlatVectorListLength { label, dim, len } => write!(
                f,
                "flat {label} data has length {len}, which is not divisible by dimension {dim}"
            ),
            Self::InconsistentFlatVectorListLength { expected, got } => write!(
                f,
                "flat velocity data has length {got}; expected {expected} to match positions"
            ),
        }
    }
}

impl std::error::Error for BoundaryError {}

pub trait ContinuousBoundary: Sync {
    fn dim(&self) -> usize;

    fn apply_position(&self, r: &mut [f64]) -> Result<(), BoundaryError>;

    fn apply_position_velocity(&self, r: &mut [f64], v: &mut [f64]) -> Result<(), BoundaryError> {
        validate_vector_len("velocity", self.dim(), v.len())?;
        let mut flip_mask = vec![0; self.dim()];
        self.apply_position_with_velocity_flip_mask(r, &mut flip_mask)?;
        for (velocity, &flip) in v.iter_mut().zip(flip_mask.iter()) {
            if flip == 1 {
                *velocity = -*velocity;
            }
        }
        Ok(())
    }

    fn apply_positions(&self, positions: &mut [f64]) -> Result<(), BoundaryError> {
        validate_flat_vector_list("positions", self.dim(), positions.len())?;
        let min_vectors_per_job = parallel_chunk_len(positions.len() / self.dim()).unwrap_or(1);
        positions
            .par_chunks_mut(self.dim())
            .with_min_len(min_vectors_per_job)
            .try_for_each(|r| self.apply_position(r))
    }

    fn apply_positions_velocities(
        &self,
        positions: &mut [f64],
        velocities: &mut [f64],
    ) -> Result<(), BoundaryError> {
        validate_flat_vector_list("positions", self.dim(), positions.len())?;
        validate_flat_vector_list("velocities", self.dim(), velocities.len())?;
        if positions.len() != velocities.len() {
            return Err(BoundaryError::InconsistentFlatVectorListLength {
                expected: positions.len(),
                got: velocities.len(),
            });
        }

        let min_vectors_per_job = parallel_chunk_len(positions.len() / self.dim()).unwrap_or(1);
        positions
            .par_chunks_mut(self.dim())
            .zip(velocities.par_chunks_mut(self.dim()))
            .with_min_len(min_vectors_per_job)
            .try_for_each(|(r, v)| self.apply_position_velocity(r, v))
    }

    fn apply_position_with_velocity_flip_mask(
        &self,
        r: &mut [f64],
        flip_mask: &mut [u8],
    ) -> Result<(), BoundaryError> {
        validate_vector_len("velocity_flip_mask", self.dim(), flip_mask.len())?;
        flip_mask.fill(0);
        self.apply_position(r)
    }
}

fn validate_bounds(min: &[f64], max: &[f64]) -> Result<(), BoundaryError> {
    if min.len() != max.len() {
        return Err(BoundaryError::InvalidVectorDimension {
            label: "bounds",
            expected: min.len(),
            got: max.len(),
        });
    }
    for d in 0..min.len() {
        if !min[d].is_finite() || !max[d].is_finite() || max[d] <= min[d] {
            return Err(BoundaryError::InvalidBounds {
                axis: d,
                min: min[d],
                max: max[d],
            });
        }
    }
    Ok(())
}

#[inline]
fn validate_vector_len(
    label: &'static str,
    expected: usize,
    got: usize,
) -> Result<(), BoundaryError> {
    if expected != got {
        return Err(BoundaryError::InvalidVectorDimension {
            label,
            expected,
            got,
        });
    }
    Ok(())
}

#[inline]
fn validate_flat_vector_list(
    label: &'static str,
    dim: usize,
    len: usize,
) -> Result<(), BoundaryError> {
    if !len.is_multiple_of(dim) {
        return Err(BoundaryError::InvalidFlatVectorListLength { label, dim, len });
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct PeriodicBox {
    min: Vec<f64>,
    max: Vec<f64>,
}

impl PeriodicBox {
    /// Construct a periodic box from per-axis lower and upper bounds.
    ///
    /// A coordinate below `min[d]` re-enters from `max[d]`; a coordinate at or
    /// above `max[d]` re-enters from `min[d]`. Velocity is unchanged because a
    /// periodic boundary identifies opposite faces of the same physical space.
    pub fn new(min: &[f64], max: &[f64]) -> Result<Self, BoundaryError> {
        validate_bounds(min, max)?;
        Ok(Self {
            min: min.to_vec(),
            max: max.to_vec(),
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.min.len()
    }

    #[inline]
    pub fn min(&self) -> &[f64] {
        &self.min
    }

    #[inline]
    pub fn max(&self) -> &[f64] {
        &self.max
    }
}

impl ContinuousBoundary for PeriodicBox {
    #[inline]
    fn dim(&self) -> usize {
        self.dim()
    }

    fn apply_position(&self, r: &mut [f64]) -> Result<(), BoundaryError> {
        validate_vector_len("position", self.dim(), r.len())?;
        for (d, x) in r.iter_mut().enumerate() {
            if !x.is_finite() {
                continue;
            }
            let lo = self.min[d];
            let hi = self.max[d];
            let w = hi - lo;
            *x = lo + (*x - lo).rem_euclid(w);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ClampBox {
    min: Vec<f64>,
    max: Vec<f64>,
}

impl ClampBox {
    /// Construct a clamping box from per-axis lower and upper bounds.
    ///
    /// A coordinate below the box is set to the lower face; a coordinate above
    /// the box is set to the upper face. Velocity is unchanged. This is useful
    /// for hard positional constraints or diagnostic state cleanup, but it does
    /// not model an elastic collision.
    pub fn new(min: &[f64], max: &[f64]) -> Result<Self, BoundaryError> {
        validate_bounds(min, max)?;
        Ok(Self {
            min: min.to_vec(),
            max: max.to_vec(),
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.min.len()
    }

    #[inline]
    pub fn min(&self) -> &[f64] {
        &self.min
    }

    #[inline]
    pub fn max(&self) -> &[f64] {
        &self.max
    }
}

impl ContinuousBoundary for ClampBox {
    #[inline]
    fn dim(&self) -> usize {
        self.dim()
    }

    fn apply_position(&self, r: &mut [f64]) -> Result<(), BoundaryError> {
        validate_vector_len("position", self.dim(), r.len())?;
        for (d, x) in r.iter_mut().enumerate() {
            *x = x.clamp(self.min[d], self.max[d]);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ReflectBox {
    min: Vec<f64>,
    max: Vec<f64>,
}

impl ReflectBox {
    /// Construct a reflecting box from per-axis lower and upper bounds.
    ///
    /// A coordinate outside the box is mirrored back by reflecting through the
    /// box faces. The matching velocity component changes sign only when the
    /// unfolded trajectory crosses an odd number of faces. Large overshoots are
    /// handled by repeated reflection rather than by a single clamp.
    pub fn new(min: &[f64], max: &[f64]) -> Result<Self, BoundaryError> {
        validate_bounds(min, max)?;
        Ok(Self {
            min: min.to_vec(),
            max: max.to_vec(),
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.min.len()
    }

    #[inline]
    pub fn min(&self) -> &[f64] {
        &self.min
    }

    #[inline]
    pub fn max(&self) -> &[f64] {
        &self.max
    }
}

impl ContinuousBoundary for ReflectBox {
    #[inline]
    fn dim(&self) -> usize {
        self.dim()
    }

    fn apply_position(&self, r: &mut [f64]) -> Result<(), BoundaryError> {
        let mut flip_mask = vec![0; self.dim()];
        self.apply_position_with_velocity_flip_mask(r, &mut flip_mask)
    }

    fn apply_position_with_velocity_flip_mask(
        &self,
        r: &mut [f64],
        flip_mask: &mut [u8],
    ) -> Result<(), BoundaryError> {
        validate_vector_len("position", self.dim(), r.len())?;
        validate_vector_len("velocity_flip_mask", self.dim(), flip_mask.len())?;
        flip_mask.fill(0);

        for d in 0..self.dim() {
            let x = r[d];
            if !x.is_finite() {
                continue;
            }

            let lo = self.min[d];
            let hi = self.max[d];
            if !(x < lo || x > hi) {
                continue;
            }

            let w = hi - lo;
            let y = (x - lo).rem_euclid(2.0 * w);
            r[d] = if y <= w { lo + y } else { hi - (y - w) };

            let flips = if x < lo {
                ((lo - x) / w).ceil() as i64
            } else {
                ((x - hi) / w).ceil() as i64
            };
            if flips & 1 == 1 {
                flip_mask[d] = 1;
            }
        }
        Ok(())
    }
}

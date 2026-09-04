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
                "boundary bounds on axis {axis} must be finite with min < max and a representable required span; got min={min}, max={max}"
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

    /// Whether correctly dimensioned vectors can still produce an error.
    ///
    /// The conservative default lets particle adapters stage fallible custom
    /// transformations and commit only on success. Override with `false` only
    /// when both position and position/velocity methods cannot fail after shape
    /// validation. This is an error-atomicity contract, not a memory-safety one.
    fn may_fail_after_validation(&self) -> bool {
        true
    }

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
    if min.is_empty() {
        return Err(BoundaryError::InvalidVectorDimension {
            label: "bounds",
            expected: 1,
            got: 0,
        });
    }
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
    if dim == 0 || !len.is_multiple_of(dim) {
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
    /// Construct a nonempty periodic box with finite, representable axis spans.
    ///
    /// A coordinate below `min[d]` re-enters from `max[d]`; a coordinate at or
    /// above `max[d]` re-enters from `min[d]`. Velocity is unchanged because a
    /// periodic boundary identifies opposite faces of the same physical space.
    pub fn new(min: &[f64], max: &[f64]) -> Result<Self, BoundaryError> {
        validate_bounds(min, max)?;
        for axis in 0..min.len() {
            if !(max[axis] - min[axis]).is_finite() {
                return Err(BoundaryError::InvalidBounds {
                    axis,
                    min: min[axis],
                    max: max[axis],
                });
            }
        }
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
    fn may_fail_after_validation(&self) -> bool {
        false
    }

    #[inline]
    fn dim(&self) -> usize {
        self.dim()
    }

    fn apply_position_velocity(&self, r: &mut [f64], v: &mut [f64]) -> Result<(), BoundaryError> {
        validate_vector_len("velocity", self.dim(), v.len())?;
        self.apply_position(r)
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
            *x = lo + shifted_remainder(*x, lo, w);
            if *x >= hi {
                *x = lo;
            }
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
    fn may_fail_after_validation(&self) -> bool {
        false
    }

    #[inline]
    fn dim(&self) -> usize {
        self.dim()
    }

    fn apply_position_velocity(&self, r: &mut [f64], v: &mut [f64]) -> Result<(), BoundaryError> {
        validate_vector_len("velocity", self.dim(), v.len())?;
        self.apply_position(r)
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
    /// Construct a nonempty reflecting box with finite doubled axis spans.
    ///
    /// A coordinate outside the box is mirrored back by reflecting through the
    /// box faces. The matching velocity component changes sign only when the
    /// unfolded trajectory crosses an odd number of faces. Large overshoots are
    /// handled by a remainder fold without per-position allocation. Nonfinite
    /// coordinates pass through unchanged; velocity flips follow wall parity.
    pub fn new(min: &[f64], max: &[f64]) -> Result<Self, BoundaryError> {
        validate_bounds(min, max)?;
        for axis in 0..min.len() {
            if !(2.0 * (max[axis] - min[axis])).is_finite() {
                return Err(BoundaryError::InvalidBounds {
                    axis,
                    min: min[axis],
                    max: max[axis],
                });
            }
        }
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
    fn may_fail_after_validation(&self) -> bool {
        false
    }

    #[inline]
    fn dim(&self) -> usize {
        self.dim()
    }

    fn apply_position(&self, r: &mut [f64]) -> Result<(), BoundaryError> {
        validate_vector_len("position", self.dim(), r.len())?;
        for (axis, value) in r.iter_mut().enumerate() {
            *value = reflect(*value, self.min[axis], self.max[axis]).0;
        }
        Ok(())
    }

    fn apply_position_velocity(&self, r: &mut [f64], v: &mut [f64]) -> Result<(), BoundaryError> {
        validate_vector_len("position", self.dim(), r.len())?;
        validate_vector_len("velocity", self.dim(), v.len())?;
        for (axis, (position, velocity)) in r.iter_mut().zip(v).enumerate() {
            let (value, flip) = reflect(*position, self.min[axis], self.max[axis]);
            *position = value;
            if flip {
                *velocity = -*velocity;
            }
        }
        Ok(())
    }

    fn apply_position_with_velocity_flip_mask(
        &self,
        r: &mut [f64],
        flip_mask: &mut [u8],
    ) -> Result<(), BoundaryError> {
        validate_vector_len("position", self.dim(), r.len())?;
        validate_vector_len("velocity_flip_mask", self.dim(), flip_mask.len())?;
        for (axis, (position, mask)) in r.iter_mut().zip(flip_mask).enumerate() {
            let (value, flip) = reflect(*position, self.min[axis], self.max[axis]);
            *position = value;
            *mask = u8::from(flip);
        }
        Ok(())
    }
}

/// Avoids overflow in translating a finite coordinate far from the origin.
fn shifted_remainder(value: f64, origin: f64, period: f64) -> f64 {
    let shifted = value - origin;
    if shifted.is_finite() {
        shifted.rem_euclid(period)
    } else {
        (value.rem_euclid(period) - origin.rem_euclid(period)).rem_euclid(period)
    }
}

/// Folds the unfolded coordinate and reports reflection parity without an
/// integer crossing count. Exact wall landings retain the existing convention.
fn reflect(value: f64, low: f64, high: f64) -> (f64, bool) {
    if !value.is_finite() || (low..=high).contains(&value) {
        return (value, false);
    }
    let width = high - low;
    let offset = shifted_remainder(value, low, 2.0 * width);
    let folded = if offset <= width {
        low + offset
    } else {
        high - (offset - width)
    };
    let flip = if value < low {
        offset >= width
    } else {
        offset > width || offset == 0.0
    };
    (folded.clamp(low, high), flip)
}

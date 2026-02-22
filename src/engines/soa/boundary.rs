/*!
General boundary-condition tools for `engines::soa`.
*/

use super::phys_obj::{PhysObj, PhysObjError};

/// Errors returned by boundary policies.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryError {
    /// Bounds along an axis must satisfy `max > min` with finite endpoints.
    InvalidBounds {
        axis: usize,
        min: f64,
        max: f64,
    },
    /// Lifted error from `PhysObj` accessors.
    PhysObj(PhysObjError),
}

impl From<PhysObjError> for BoundaryError {
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `from` logic for this module.
    /// - Parameters:
    ///   - `value` (`PhysObjError`): Value provided by caller for write/update behavior.
    fn from(value: PhysObjError) -> Self {
        Self::PhysObj(value)
    }
}

/// Generic boundary-condition interface.
pub trait Boundary<const D: usize> {
    /// Apply boundary transform to one particle state.
    /// Annotation:
    /// - Purpose: Applies `particle` behavior.
    /// - Parameters:
    ///   - `pos` (`&mut [f64]`): Parameter of type `&mut [f64]` used by `apply_particle`.
    ///   - `vel` (`&mut [f64]`): Parameter of type `&mut [f64]` used by `apply_particle`.
    fn apply_particle(&self, pos: &mut [f64], vel: &mut [f64]);

    /// Apply to all alive particles.
    /// Annotation:
    /// - Purpose: Applies `all` behavior.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj<D>`): Object-state container used by this operation.
    fn apply_all(&self, objects: &mut PhysObj<D>) -> Result<(), BoundaryError> {
        let n = objects.len();
        for i in 0..n {
            if !objects.is_alive(i)? {
                continue;
            }

            // Copy-transform-writeback keeps API safe and backend-agnostic.
            let mut pos = objects.pos_of(i)?.to_vec();
            let mut vel = objects.vel_of(i)?.to_vec();
            self.apply_particle(&mut pos, &mut vel);
            objects.set_pos(i, &pos)?;
            objects.set_vel(i, &vel)?;
        }
        Ok(())
    }
}

#[inline]
fn validate_bounds<const D: usize>(min: &[f64; D], max: &[f64; D]) -> Result<(), BoundaryError> {
    for d in 0..D {
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

/// Periodic box boundary.
#[derive(Debug, Clone, Copy)]
pub struct PeriodicBox<const D: usize> {
    min: [f64; D],
    max: [f64; D],
}

impl<const D: usize> PeriodicBox<D> {
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `min` (`[f64; D]`): Parameter of type `[f64; D]` used by `new`.
    ///   - `max` (`[f64; D]`): Parameter of type `[f64; D]` used by `new`.
    pub fn new(min: [f64; D], max: [f64; D]) -> Result<Self, BoundaryError> {
        validate_bounds(&min, &max)?;
        Ok(Self { min, max })
    }
}

impl<const D: usize> Boundary<D> for PeriodicBox<D> {
    /// Annotation:
    /// - Purpose: Applies `particle` behavior.
    /// - Parameters:
    ///   - `pos` (`&mut [f64]`): Parameter of type `&mut [f64]` used by `apply_particle`.
    ///   - `_vel` (`&mut [f64]`): Parameter of type `&mut [f64]` used by `apply_particle`.
    fn apply_particle(&self, pos: &mut [f64], _vel: &mut [f64]) {
        for d in 0..D {
            let lo = self.min[d];
            let hi = self.max[d];
            let w = hi - lo;
            while pos[d] < lo {
                pos[d] += w;
            }
            while pos[d] >= hi {
                pos[d] -= w;
            }
        }
    }
}

/// Hard clamp boundary.
#[derive(Debug, Clone, Copy)]
pub struct ClampBox<const D: usize> {
    min: [f64; D],
    max: [f64; D],
}

impl<const D: usize> ClampBox<D> {
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `min` (`[f64; D]`): Parameter of type `[f64; D]` used by `new`.
    ///   - `max` (`[f64; D]`): Parameter of type `[f64; D]` used by `new`.
    pub fn new(min: [f64; D], max: [f64; D]) -> Result<Self, BoundaryError> {
        validate_bounds(&min, &max)?;
        Ok(Self { min, max })
    }
}

impl<const D: usize> Boundary<D> for ClampBox<D> {
    /// Annotation:
    /// - Purpose: Applies `particle` behavior.
    /// - Parameters:
    ///   - `pos` (`&mut [f64]`): Parameter of type `&mut [f64]` used by `apply_particle`.
    ///   - `_vel` (`&mut [f64]`): Parameter of type `&mut [f64]` used by `apply_particle`.
    fn apply_particle(&self, pos: &mut [f64], _vel: &mut [f64]) {
        for d in 0..D {
            pos[d] = pos[d].clamp(self.min[d], self.max[d]);
        }
    }
}

/// Reflective boundary.
#[derive(Debug, Clone, Copy)]
pub struct ReflectBox<const D: usize> {
    min: [f64; D],
    max: [f64; D],
}

impl<const D: usize> ReflectBox<D> {
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `min` (`[f64; D]`): Parameter of type `[f64; D]` used by `new`.
    ///   - `max` (`[f64; D]`): Parameter of type `[f64; D]` used by `new`.
    pub fn new(min: [f64; D], max: [f64; D]) -> Result<Self, BoundaryError> {
        validate_bounds(&min, &max)?;
        Ok(Self { min, max })
    }
}

impl<const D: usize> Boundary<D> for ReflectBox<D> {
    /// Annotation:
    /// - Purpose: Applies `particle` behavior.
    /// - Parameters:
    ///   - `pos` (`&mut [f64]`): Parameter of type `&mut [f64]` used by `apply_particle`.
    ///   - `vel` (`&mut [f64]`): Parameter of type `&mut [f64]` used by `apply_particle`.
    fn apply_particle(&self, pos: &mut [f64], vel: &mut [f64]) {
        for d in 0..D {
            let lo = self.min[d];
            let hi = self.max[d];

            while pos[d] < lo || pos[d] > hi {
                if pos[d] < lo {
                    pos[d] = lo + (lo - pos[d]);
                    vel[d] = -vel[d];
                }
                if pos[d] > hi {
                    pos[d] = hi - (pos[d] - hi);
                    vel[d] = -vel[d];
                }
            }
        }
    }
}

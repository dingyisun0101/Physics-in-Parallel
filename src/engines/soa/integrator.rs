/*!
General time-integration tools for `engines::soa`.

This module provides backend-agnostic integrator traits and baseline explicit
schemes that act on `PhysObj` core state (`pos`, `vel`, `acc`).
*/

use super::phys_obj::{PhysObj, PhysObjError};

/// Errors returned by integrators.
#[derive(Debug, Clone, PartialEq)]
pub enum IntegratorError {
    /// Time-step must be finite and strictly positive.
    InvalidDt { dt: f64 },
    /// Lifted error from `PhysObj` accessors.
    PhysObj(PhysObjError),
}

impl From<PhysObjError> for IntegratorError {
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `from` logic for this module.
    /// - Parameters:
    ///   - `value` (`PhysObjError`): Value provided by caller for write/update behavior.
    fn from(value: PhysObjError) -> Self {
        Self::PhysObj(value)
    }
}

/// Generic time integrator contract for SoA object states.
pub trait Integrator<const D: usize> {
    /// Advance one step using existing acceleration values in `objects.acc`.
    /// Annotation:
    /// - Purpose: Executes `step` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj<D>`): Object-state container used by this operation.
    ///   - `dt` (`f64`): Time-step used for one simulation/integration update.
    fn step(&mut self, objects: &mut PhysObj<D>, dt: f64) -> Result<(), IntegratorError>;
}

/// Explicit Euler (`r += v*dt`, `v += a*dt`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ExplicitEuler;

impl<const D: usize> Integrator<D> for ExplicitEuler {
    /// Annotation:
    /// - Purpose: Executes `step` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj<D>`): Object-state container used by this operation.
    ///   - `dt` (`f64`): Time-step used for one simulation/integration update.
    fn step(&mut self, objects: &mut PhysObj<D>, dt: f64) -> Result<(), IntegratorError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(IntegratorError::InvalidDt { dt });
        }

        let n = objects.len();
        for i in 0..n {
            if !objects.is_alive(i)? {
                continue;
            }

            let v = objects.vel_of(i)?.to_vec();
            let a = objects.acc_of(i)?.to_vec();
            let mut r = objects.pos_of(i)?.to_vec();

            for d in 0..D {
                r[d] += v[d] * dt;
            }
            objects.set_pos(i, &r)?;

            let mut v_new = v;
            for d in 0..D {
                v_new[d] += a[d] * dt;
            }
            objects.set_vel(i, &v_new)?;
        }

        Ok(())
    }
}

/// Semi-implicit Euler / Euler-Cromer (`v += a*dt`, `r += v*dt`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SemiImplicitEuler;

impl<const D: usize> Integrator<D> for SemiImplicitEuler {
    /// Annotation:
    /// - Purpose: Executes `step` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj<D>`): Object-state container used by this operation.
    ///   - `dt` (`f64`): Time-step used for one simulation/integration update.
    fn step(&mut self, objects: &mut PhysObj<D>, dt: f64) -> Result<(), IntegratorError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(IntegratorError::InvalidDt { dt });
        }

        let n = objects.len();
        for i in 0..n {
            if !objects.is_alive(i)? {
                continue;
            }

            let a = objects.acc_of(i)?.to_vec();
            let mut v = objects.vel_of(i)?.to_vec();
            for d in 0..D {
                v[d] += a[d] * dt;
            }
            objects.set_vel(i, &v)?;

            let mut r = objects.pos_of(i)?.to_vec();
            for d in 0..D {
                r[d] += v[d] * dt;
            }
            objects.set_pos(i, &r)?;
        }

        Ok(())
    }
}

/*!
General time-integration tools for massive-particle models.
*/

use rayon::prelude::*;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{ATTR_A, ATTR_ALIVE, ATTR_R, ATTR_V};

#[derive(Debug, Clone, PartialEq)]
pub enum IntegratorError {
    /// Integration step size is invalid for this solver update.
    InvalidDt {
        /// Candidate time step passed by caller.
        dt: f64,
    },
    /// Error bubbled up from underlying attribute storage.
    Attrs(AttrsError),
    /// One attribute column has unexpected per-particle vector dimension.
    InvalidAttrShape {
        /// Attribute label that failed shape validation.
        label: &'static str,
        /// Expected vector dimension for this attribute.
        expected_dim: usize,
        /// Observed vector dimension in storage.
        got_dim: usize,
    },
    /// One attribute column has inconsistent number of particles.
    InconsistentParticleCount {
        /// Attribute label that failed particle-count validation.
        label: &'static str,
        /// Expected particle count derived from canonical attributes.
        expected: usize,
        /// Observed particle count in storage.
        got: usize,
    },
}

impl From<AttrsError> for IntegratorError {
    #[inline]
    /// - Purpose: Converts an `AttrsError` into `IntegratorError`.
    /// - Parameters:
    ///   - `value` (`AttrsError`): Source error emitted by attribute storage operations.
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
}

#[inline]
/// - Purpose: Reads and validates optional alive-mask data for particle-wise updates.
/// - Parameters:
///   - `objects` (`&PhysObj`): SoA object container queried for optional `alive` attribute.
///   - `n` (`usize`): Expected particle count used to validate the alive-mask length.
fn gather_alive_flags(objects: &PhysObj, n: usize) -> Result<Option<Vec<bool>>, IntegratorError> {
    if !objects.core.contains(ATTR_ALIVE) {
        return Ok(None);
    }

    let alive = objects.core.get::<f64>(ATTR_ALIVE)?;
    if alive.dim() != 1 {
        return Err(IntegratorError::InvalidAttrShape {
            label: ATTR_ALIVE,
            expected_dim: 1,
            got_dim: alive.dim(),
        });
    }
    if alive.num_vectors() != n {
        return Err(IntegratorError::InconsistentParticleCount {
            label: ATTR_ALIVE,
            expected: n,
            got: alive.num_vectors(),
        });
    }

    let mut flags = Vec::with_capacity(n);
    for i in 0..n {
        flags.push(alive.get(i as isize, 0) > 0.0);
    }
    Ok(Some(flags))
}

pub trait Integrator {
    /// - Purpose: Advances the particle state by one time step.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj`): SoA object container holding attributes updated by this integrator.
    ///   - `dt` (`f64`): Positive finite time-step size for one integration step.
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError>;
}

/// Explicit Euler integrator marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExplicitEuler;

/// Semi-implicit Euler integrator marker type.
#[derive(Debug, Clone, Copy, Default)]
pub struct SemiImplicitEuler;

#[inline]
/// - Purpose: Shared Euler kernel that updates velocity from acceleration, then position from updated velocity.
/// - Parameters:
///   - `objects` (`&mut PhysObj`): SoA object container containing `r`, `v`, and `a` attributes to update in-place.
///   - `dt` (`f64`): Positive finite time-step size for one integration step.
fn apply_v_then_r(objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(IntegratorError::InvalidDt { dt });
    }

    let (dim, n) = {
        let v = objects.core.get::<f64>(ATTR_V)?;
        (v.dim(), v.num_vectors())
    };

    {
        let a = objects.core.get::<f64>(ATTR_A)?;
        if a.dim() != dim {
            return Err(IntegratorError::InvalidAttrShape {
                label: ATTR_A,
                expected_dim: dim,
                got_dim: a.dim(),
            });
        }
        if a.num_vectors() != n {
            return Err(IntegratorError::InconsistentParticleCount {
                label: ATTR_A,
                expected: n,
                got: a.num_vectors(),
            });
        }
    }

    {
        let r = objects.core.get::<f64>(ATTR_R)?;
        if r.dim() != dim {
            return Err(IntegratorError::InvalidAttrShape {
                label: ATTR_R,
                expected_dim: dim,
                got_dim: r.dim(),
            });
        }
        if r.num_vectors() != n {
            return Err(IntegratorError::InconsistentParticleCount {
                label: ATTR_R,
                expected: n,
                got: r.num_vectors(),
            });
        }
    }

    let alive_flags = gather_alive_flags(objects, n)?;
    let a_data: Vec<f64> = {
        let a = objects.core.get::<f64>(ATTR_A)?;
        a.as_tensor().data.clone()
    };

    let updated_v_data: Vec<f64> = {
        let v = objects.core.get_mut::<f64>(ATTR_V)?;
        let v_data = &mut v.as_tensor_mut().data;

        v_data
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(i, v_row)| {
                if let Some(flags) = &alive_flags {
                    if !flags[i] {
                        return;
                    }
                }

                let a_row = &a_data[i * dim..(i + 1) * dim];
                for k in 0..dim {
                    v_row[k] += a_row[k] * dt;
                }
            });

        v_data.clone()
    };

    {
        let r = objects.core.get_mut::<f64>(ATTR_R)?;
        let r_data = &mut r.as_tensor_mut().data;

        r_data
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(i, r_row)| {
                if let Some(flags) = &alive_flags {
                    if !flags[i] {
                        return;
                    }
                }

                let v_row = &updated_v_data[i * dim..(i + 1) * dim];
                for k in 0..dim {
                    r_row[k] += v_row[k] * dt;
                }
            });
    }

    Ok(())
}

impl Integrator for ExplicitEuler {
    /// - Purpose: Applies one explicit-Euler step to the particle attributes.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj`): SoA object container containing mutable state vectors.
    ///   - `dt` (`f64`): Positive finite time-step size for one integration step.
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError> {
        apply_v_then_r(objects, dt)
    }
}

impl Integrator for SemiImplicitEuler {
    /// - Purpose: Applies one semi-implicit Euler step to the particle attributes.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj`): SoA object container containing mutable state vectors.
    ///   - `dt` (`f64`): Positive finite time-step size for one integration step.
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError> {
        apply_v_then_r(objects, dt)
    }
}

/*!
Time-integration schemes for canonical massive-particle state.

Purpose:
Integrators advance `r` and `v` in a `PhysObj` using the canonical attributes
`ATTR_R`, `ATTR_V`, and `ATTR_A`.

Semantics:
`ExplicitEuler` applies `r_next = r + v_old * dt` and
`v_next = v + a * dt`.

`SemiImplicitEuler` applies `v_next = v + a * dt` and
`r_next = r + v_next * dt`.

Both integrators skip dead particles (`alive == false`) and rigid particles.
Acceleration is not cleared after integration; force/acceleration management is
left to the caller.
*/

use rayon::prelude::*;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{ATTR_A, ATTR_R, ATTR_V, ParticleSelection};
use crate::models::particles::state::{
    ParticleMasks, ParticleStateError, gather_masks, validate_vector_attr_f64,
};

/// Errors returned by time integrators.
#[derive(Debug, Clone, PartialEq)]
pub enum IntegratorError {
    /// Integration step size is not finite or is not strictly positive.
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
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
}

impl From<ParticleStateError> for IntegratorError {
    fn from(value: ParticleStateError) -> Self {
        match value {
            ParticleStateError::Attrs(err) => Self::Attrs(err),
            ParticleStateError::InvalidAttrShape {
                label,
                expected_dim,
                got_dim,
            } => Self::InvalidAttrShape {
                label,
                expected_dim,
                got_dim,
            },
            ParticleStateError::InconsistentParticleCount {
                label,
                expected,
                got,
            } => Self::InconsistentParticleCount {
                label,
                expected,
                got,
            },
        }
    }
}

/// Time-integrator contract.
pub trait Integrator {
    /// Advances particle state by one time step.
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError>;
}

/// Explicit Euler integrator.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExplicitEuler;

/// Semi-implicit Euler integrator.
#[derive(Debug, Clone, Copy, Default)]
pub struct SemiImplicitEuler;

#[derive(Debug, Clone)]
struct IntegratorContext {
    dim: usize,
    masks: ParticleMasks,
}

fn validate_dt(dt: f64) -> Result<(), IntegratorError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(IntegratorError::InvalidDt { dt });
    }
    Ok(())
}

fn validate_core_shapes(objects: &PhysObj) -> Result<IntegratorContext, IntegratorError> {
    let (dim, n) = {
        let v = objects.core.get::<f64>(ATTR_V)?;
        (v.dim(), v.num_vectors())
    };

    validate_vector_attr_f64(objects, ATTR_A, dim, n)?;
    validate_vector_attr_f64(objects, ATTR_R, dim, n)?;

    Ok(IntegratorContext {
        dim,
        masks: gather_masks(objects, n, ParticleSelection::AliveOnly)?,
    })
}

fn should_skip(ctx: &IntegratorContext, i: usize) -> bool {
    ctx.masks.should_skip(i)
}

fn apply_explicit_euler(objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError> {
    validate_dt(dt)?;
    let ctx = validate_core_shapes(objects)?;

    let a_data = objects.core.get::<f64>(ATTR_A)?.as_tensor().data.clone();
    let v_old = objects.core.get::<f64>(ATTR_V)?.as_tensor().data.clone();
    let mut v_new = v_old.clone();

    v_new
        .par_chunks_mut(ctx.dim)
        .enumerate()
        .for_each(|(i, v_row)| {
            if should_skip(&ctx, i) {
                return;
            }

            let a_row = &a_data[i * ctx.dim..(i + 1) * ctx.dim];
            for k in 0..ctx.dim {
                v_row[k] += a_row[k] * dt;
            }
        });

    {
        let r = objects.core.get_mut::<f64>(ATTR_R)?;
        r.as_tensor_mut()
            .data
            .par_chunks_mut(ctx.dim)
            .enumerate()
            .for_each(|(i, r_row)| {
                if should_skip(&ctx, i) {
                    return;
                }

                let v_row = &v_old[i * ctx.dim..(i + 1) * ctx.dim];
                for k in 0..ctx.dim {
                    r_row[k] += v_row[k] * dt;
                }
            });
    }

    objects
        .core
        .get_mut::<f64>(ATTR_V)?
        .as_tensor_mut()
        .data
        .copy_from_slice(&v_new);

    Ok(())
}

fn apply_semi_implicit_euler(objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError> {
    validate_dt(dt)?;
    let ctx = validate_core_shapes(objects)?;

    let a_data = objects.core.get::<f64>(ATTR_A)?.as_tensor().data.clone();
    let updated_v_data = {
        let v = objects.core.get_mut::<f64>(ATTR_V)?;
        let v_data = &mut v.as_tensor_mut().data;

        v_data
            .par_chunks_mut(ctx.dim)
            .enumerate()
            .for_each(|(i, v_row)| {
                if should_skip(&ctx, i) {
                    return;
                }

                let a_row = &a_data[i * ctx.dim..(i + 1) * ctx.dim];
                for k in 0..ctx.dim {
                    v_row[k] += a_row[k] * dt;
                }
            });

        v_data.clone()
    };

    {
        let r = objects.core.get_mut::<f64>(ATTR_R)?;
        r.as_tensor_mut()
            .data
            .par_chunks_mut(ctx.dim)
            .enumerate()
            .for_each(|(i, r_row)| {
                if should_skip(&ctx, i) {
                    return;
                }

                let v_row = &updated_v_data[i * ctx.dim..(i + 1) * ctx.dim];
                for k in 0..ctx.dim {
                    r_row[k] += v_row[k] * dt;
                }
            });
    }

    Ok(())
}

impl Integrator for ExplicitEuler {
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError> {
        apply_explicit_euler(objects, dt)
    }
}

impl Integrator for SemiImplicitEuler {
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError> {
        apply_semi_implicit_euler(objects, dt)
    }
}

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

use core::fmt;

use crate::threading::try_for_each_pair_chunk_mut;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{ATTR_A, ATTR_R, ATTR_V, ParticleSelection};
use crate::models::particles::state::{
    BorrowedMasks, ParticleStateError, mask_labels, validate_vector_attr_f64,
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
    State(ParticleStateError),
}

impl From<AttrsError> for IntegratorError {
    fn from(value: AttrsError) -> Self {
        Self::State(value.into())
    }
}

impl From<ParticleStateError> for IntegratorError {
    fn from(value: ParticleStateError) -> Self {
        Self::State(value)
    }
}

impl fmt::Display for IntegratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDt { dt } => {
                write!(
                    f,
                    "integration time step must be finite and positive; got {dt}"
                )
            }
            Self::State(error) => write!(f, "invalid particle state: {error}"),
        }
    }
}

impl std::error::Error for IntegratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(error) => Some(error),
            _ => None,
        }
    }
}

/// Time-integrator contract.
pub trait Integrator: Send {
    /// Advances particle state by one time step.
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError>;
}

/// Explicit Euler integrator.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExplicitEuler;

/// Semi-implicit Euler integrator.
#[derive(Debug, Clone, Copy, Default)]
pub struct SemiImplicitEuler;

/// Validates once and fuses each row's position/velocity update. Dense columns
/// are borrowed; sparse mutations use the vector-list staging path. The explicit
/// variant saves old velocity before writing either result.
fn apply_euler(objects: &mut PhysObj, dt: f64, semi_implicit: bool) -> Result<(), IntegratorError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(IntegratorError::InvalidDt { dt });
    }
    let v = objects.core.get::<f64>(ATTR_V)?;
    let (dim, n) = (v.dim(), v.num_vectors());
    validate_vector_attr_f64(objects, ATTR_A, dim, n)?;
    validate_vector_attr_f64(objects, ATTR_R, dim, n)?;
    let flags = mask_labels(objects, ParticleSelection::AliveOnly);
    let ([r, v, a], flags) = objects
        .core
        .get_mixed_mut::<f64, u8, 3, 2>([ATTR_R, ATTR_V, ATTR_A], flags)?;
    let masks = BorrowedMasks::new(flags, n)?;
    let acceleration = a.borrow_values();
    r.edit_values(|positions| {
        v.edit_values(|velocities| {
            try_for_each_pair_chunk_mut(
                positions,
                velocities,
                dim,
                |start, positions, velocities| {
                    for (row, (r, v)) in positions
                        .chunks_exact_mut(dim)
                        .zip(velocities.chunks_exact_mut(dim))
                        .enumerate()
                    {
                        let index = start / dim + row;
                        if masks.should_skip(index) {
                            continue;
                        }
                        let a = &acceleration[index * dim..][..dim];
                        for ((r, v), &a) in r.iter_mut().zip(v).zip(a) {
                            let old = *v;
                            *v += a * dt;
                            *r += if semi_implicit { *v * dt } else { old * dt };
                        }
                    }
                    Ok::<_, std::convert::Infallible>(())
                },
            )
            .expect("infallible validated integration");
        })
    });
    Ok(())
}

impl Integrator for ExplicitEuler {
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError> {
        apply_euler(objects, dt, false)
    }
}

impl Integrator for SemiImplicitEuler {
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), IntegratorError> {
        apply_euler(objects, dt, true)
    }
}

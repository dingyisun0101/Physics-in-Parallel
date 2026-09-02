/*!
Read-only particle observables.

Purpose:
This module computes scalar summaries from canonical massive-particle state
stored in `PhysObj`. Observers do not mutate particle state. They validate the
attribute shapes and numeric values they depend on, then return a physically
meaningful aggregate such as total kinetic energy or kinetic temperature.

Conventions:
`KineticEnergyObserver` expects inverse mass `m_inv = 1 / m` and computes
`0.5 * |v|^2 / m_inv` for each included particle. This requires strictly
positive inverse mass. Zero inverse mass represents an infinite-mass limit, for
which kinetic energy is not finite under this convention, so it is reported as
an invalid state.

With `ParticleSelection::AliveOnly`, dead particles do not contribute to
observables. `ParticleSelection::All` intentionally ignores `ATTR_ALIVE` and is
intended for debugging or inspecting all allocated slots.
*/

use core::fmt;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{ATTR_M_INV, ATTR_V, ParticleSelection};
use crate::models::particles::state::{
    ParticleMasks, ParticleStateError, gather_alive_flags, gather_inverse_mass,
};
use rayon::prelude::*;

/// Errors returned by particle observers.
#[derive(Debug, Clone, PartialEq)]
pub enum ObserveError {
    State(ParticleStateError),
    /// Numeric state violates an observer precondition.
    InvalidState {
        /// Name of the state field with invalid value.
        field: &'static str,
        /// Invalid numeric value encountered by the observer.
        value: f64,
    },
}

impl From<AttrsError> for ObserveError {
    fn from(value: AttrsError) -> Self {
        Self::State(value.into())
    }
}

impl From<ParticleStateError> for ObserveError {
    fn from(value: ParticleStateError) -> Self {
        Self::State(value)
    }
}

impl fmt::Display for ObserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(error) => write!(f, "invalid particle state: {error}"),
            Self::InvalidState { field, value } => {
                write!(f, "observer found invalid `{field}` value {value}")
            }
        }
    }
}

impl std::error::Error for ObserveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct KineticContext {
    dim: usize,
    n: usize,
    v_data: Vec<f64>,
    m_inv_values: Vec<f64>,
    masks: ParticleMasks,
    selection: ParticleSelection,
}

fn gather_kinetic_context(
    objects: &PhysObj,
    selection: ParticleSelection,
) -> Result<KineticContext, ObserveError> {
    let (dim, n, v_data) = {
        let v = objects.core.get::<f64>(ATTR_V)?;
        (v.dim(), v.num_vectors(), v.as_tensor().data.clone())
    };

    Ok(KineticContext {
        dim,
        n,
        v_data,
        m_inv_values: gather_inverse_mass(objects, n)?,
        masks: ParticleMasks {
            alive: gather_alive_flags(objects, n, selection)?,
            rigid: None,
        },
        selection,
    })
}

fn kinetic_energy_from_context(ctx: &KineticContext) -> Result<f64, ObserveError> {
    (0..ctx.n)
        .into_par_iter()
        .map(|i| -> Result<f64, ObserveError> {
            if !ctx.masks.is_included(ctx.selection, i) {
                return Ok(0.0);
            }

            let m_inv_i = ctx.m_inv_values[i];
            if !m_inv_i.is_finite() || m_inv_i <= 0.0 {
                return Err(ObserveError::InvalidState {
                    field: ATTR_M_INV,
                    value: m_inv_i,
                });
            }

            let row = &ctx.v_data[i * ctx.dim..(i + 1) * ctx.dim];
            let mut v2 = 0.0;
            for &component in row {
                if !component.is_finite() {
                    return Err(ObserveError::InvalidState {
                        field: ATTR_V,
                        value: component,
                    });
                }
                v2 += component * component;
            }
            Ok(0.5 * v2 / m_inv_i)
        })
        .try_reduce(|| 0.0, |a, b| Ok(a + b))
}

/// Computes one observable from the current particle state.
pub trait Observer: Send + Sync {
    type Output;

    /// Computes one observable from `objects`.
    fn observe(&self, objects: &PhysObj) -> Result<Self::Output, ObserveError>;
}

/// Total kinetic-energy observer.
#[derive(Debug, Clone, Copy)]
pub struct KineticEnergyObserver {
    /// Which particle subset contributes to kinetic energy.
    selection: ParticleSelection,
}

impl KineticEnergyObserver {
    /// Builds a kinetic-energy observer with explicit particle-selection policy.
    pub fn new(selection: ParticleSelection) -> Self {
        Self { selection }
    }

    /// Returns the particle subset included by this observer.
    pub fn selection(&self) -> ParticleSelection {
        self.selection
    }
}

impl Default for KineticEnergyObserver {
    fn default() -> Self {
        Self {
            selection: ParticleSelection::AliveOnly,
        }
    }
}

impl Observer for KineticEnergyObserver {
    type Output = f64;

    fn observe(&self, objects: &PhysObj) -> Result<Self::Output, ObserveError> {
        let ctx = gather_kinetic_context(objects, self.selection)?;
        kinetic_energy_from_context(&ctx)
    }
}

/// Kinetic-temperature observer using `2 * KE / degrees_of_freedom`.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureObserver {
    /// Which particle subset contributes to temperature.
    selection: ParticleSelection,
}

impl TemperatureObserver {
    /// Builds a temperature observer with explicit particle-selection policy.
    pub fn new(selection: ParticleSelection) -> Self {
        Self { selection }
    }

    /// Returns the particle subset included by this observer.
    pub fn selection(&self) -> ParticleSelection {
        self.selection
    }
}

impl Default for TemperatureObserver {
    fn default() -> Self {
        Self {
            selection: ParticleSelection::AliveOnly,
        }
    }
}

impl Observer for TemperatureObserver {
    type Output = f64;

    fn observe(&self, objects: &PhysObj) -> Result<Self::Output, ObserveError> {
        let ctx = gather_kinetic_context(objects, self.selection)?;
        let ke = kinetic_energy_from_context(&ctx)?;
        let count = ctx.masks.included_count(ctx.selection, ctx.n);

        if count == 0 || ctx.dim == 0 {
            return Ok(0.0);
        }

        Ok((2.0 * ke) / ((count * ctx.dim) as f64))
    }
}

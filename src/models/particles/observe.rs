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
use crate::models::particles::state::{BorrowedMasks, ParticleStateError, validate_scalar_shape};

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

/// Combined kinetic observation from one validated pass over included particles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KineticSummary {
    /// Total `0.5 * |v|² / m_inv`, including rigid particles if selected.
    pub energy: f64,
    /// `2 * energy / (particle_count * dimension)`, or zero for no particles.
    pub temperature: f64,
    /// Number of selected particles (rigidity does not remove degrees of freedom).
    pub particle_count: usize,
}

/// Computes energy, temperature and population together without repeated gathers.
///
/// Dense columns are borrowed without allocation; sparse columns stage logical
/// values. Mass/velocity checks and accumulation follow particle/component order
/// on the caller thread. Every included inverse mass must be finite and positive;
/// every included velocity component must be finite. No state is mutated.
pub fn kinetic_summary(
    objects: &PhysObj,
    selection: ParticleSelection,
) -> Result<KineticSummary, ObserveError> {
    let velocity = objects.core.get::<f64>(ATTR_V)?;
    let (dim, n) = (velocity.dim(), velocity.num_vectors());
    let mass = objects.core.get::<f64>(ATTR_M_INV)?;
    validate_scalar_shape(ATTR_M_INV, mass.dim(), mass.num_vectors(), n)?;
    let alive = if !selection.includes_dead() && objects.core.contains(super::attrs::ATTR_ALIVE) {
        Some(objects.core.get::<u8>(super::attrs::ATTR_ALIVE)?)
    } else {
        None
    };
    let masks = BorrowedMasks::new([alive, None], n)?;
    let velocity = velocity.borrow_values();
    let mass = mass.borrow_values();
    let mut energy = 0.0;
    let mut count = 0;
    for (index, row) in velocity.chunks_exact(dim).enumerate() {
        if !masks.alive(index) {
            continue;
        }
        let mass = mass[index];
        if !mass.is_finite() || mass <= 0.0 {
            return Err(ObserveError::InvalidState {
                field: ATTR_M_INV,
                value: mass,
            });
        }
        let mut squared = 0.0;
        for &component in row {
            if !component.is_finite() {
                return Err(ObserveError::InvalidState {
                    field: ATTR_V,
                    value: component,
                });
            }
            squared += component * component;
        }
        energy += 0.5 * squared / mass;
        count += 1;
    }
    Ok(KineticSummary {
        energy,
        temperature: if count == 0 {
            0.0
        } else {
            2.0 * energy / (count * dim) as f64
        },
        particle_count: count,
    })
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
        Ok(kinetic_summary(objects, self.selection)?.energy)
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
        Ok(kinetic_summary(objects, self.selection)?.temperature)
    }
}

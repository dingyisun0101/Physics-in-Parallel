/*!
Read-only particle observables and reducers.

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

use rayon::prelude::*;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{
    ATTR_ALIVE, ATTR_M_INV, ATTR_V, ParticleSelection, is_alive_value,
};

/// Errors returned by particle observers.
#[derive(Debug, Clone, PartialEq)]
pub enum ObserveError {
    /// Error bubbled up from underlying attribute storage.
    Attrs(AttrsError),
    /// Numeric state violates an observer precondition.
    InvalidState {
        /// Name of the state field with invalid value.
        field: &'static str,
        /// Invalid numeric value encountered by the observer.
        value: f64,
    },
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

impl From<AttrsError> for ObserveError {
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
}

/// Computes one observable from the current particle state.
pub trait Observer {
    type Output;

    /// Computes one observable from `objects`.
    fn observe(&self, objects: &PhysObj) -> Result<Self::Output, ObserveError>;
}

/// Reduces a batch of observed values into one aggregate value.
pub trait Reducer<T> {
    /// Reduces `values` into one aggregate.
    fn reduce(&self, values: &[T]) -> T;
}

/// Arithmetic mean reducer for scalar observations.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanReducer;

impl Reducer<f64> for MeanReducer {
    fn reduce(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / (values.len() as f64)
    }
}

/// Total kinetic-energy observer.
#[derive(Debug, Clone, Copy)]
pub struct KineticEnergyObserver {
    /// Which particle subset contributes to kinetic energy.
    pub selection: ParticleSelection,
}

impl KineticEnergyObserver {
    /// Builds a kinetic-energy observer with explicit particle-selection policy.
    pub fn new(selection: ParticleSelection) -> Self {
        Self { selection }
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
        let (dim, n, v_data) = {
            let v = objects.core.get::<f64>(ATTR_V)?;
            (v.dim(), v.num_vectors(), v.as_tensor().data.clone())
        };

        let m_inv_values = gather_inverse_mass(objects, n)?;
        let alive_flags = gather_alive_flags(objects, n)?;

        (0..n)
            .into_par_iter()
            .map(|i| -> Result<f64, ObserveError> {
                if !is_included(self.selection, &alive_flags, i) {
                    return Ok(0.0);
                }

                let m_inv_i = m_inv_values[i];
                if !m_inv_i.is_finite() || m_inv_i <= 0.0 {
                    return Err(ObserveError::InvalidState {
                        field: ATTR_M_INV,
                        value: m_inv_i,
                    });
                }

                let row = &v_data[i * dim..(i + 1) * dim];
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
}

/// Kinetic-temperature observer using `2 * KE / degrees_of_freedom`.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureObserver {
    /// Which particle subset contributes to temperature.
    pub selection: ParticleSelection,
}

impl TemperatureObserver {
    /// Builds a temperature observer with explicit particle-selection policy.
    pub fn new(selection: ParticleSelection) -> Self {
        Self { selection }
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
        let ke = KineticEnergyObserver::new(self.selection).observe(objects)?;
        let (dim, n) = {
            let v = objects.core.get::<f64>(ATTR_V)?;
            (v.dim(), v.num_vectors())
        };
        let alive_flags = gather_alive_flags(objects, n)?;
        let count = included_particles(self.selection, &alive_flags, n);

        if count == 0 || dim == 0 {
            return Ok(0.0);
        }

        Ok((2.0 * ke) / ((count * dim) as f64))
    }
}

fn gather_inverse_mass(objects: &PhysObj, n: usize) -> Result<Vec<f64>, ObserveError> {
    let m_inv = objects.core.get::<f64>(ATTR_M_INV)?;
    validate_attr_shape(ATTR_M_INV, m_inv.dim(), 1)?;
    validate_attr_count(ATTR_M_INV, m_inv.num_vectors(), n)?;

    Ok((0..n).map(|i| m_inv.get(i as isize, 0)).collect())
}

fn gather_alive_flags(objects: &PhysObj, n: usize) -> Result<Option<Vec<bool>>, ObserveError> {
    if !objects.core.contains(ATTR_ALIVE) {
        return Ok(None);
    }

    let alive = objects.core.get::<u8>(ATTR_ALIVE)?;
    validate_attr_shape(ATTR_ALIVE, alive.dim(), 1)?;
    validate_attr_count(ATTR_ALIVE, alive.num_vectors(), n)?;

    Ok(Some(
        (0..n)
            .map(|i| is_alive_value(alive.get(i as isize, 0)))
            .collect(),
    ))
}

fn validate_attr_shape(
    label: &'static str,
    got_dim: usize,
    expected_dim: usize,
) -> Result<(), ObserveError> {
    if got_dim != expected_dim {
        return Err(ObserveError::InvalidAttrShape {
            label,
            expected_dim,
            got_dim,
        });
    }
    Ok(())
}

fn validate_attr_count(
    label: &'static str,
    got: usize,
    expected: usize,
) -> Result<(), ObserveError> {
    if got != expected {
        return Err(ObserveError::InconsistentParticleCount {
            label,
            expected,
            got,
        });
    }
    Ok(())
}

fn is_included(selection: ParticleSelection, alive_flags: &Option<Vec<bool>>, i: usize) -> bool {
    if selection.includes_dead() {
        return true;
    }
    alive_flags.as_ref().is_none_or(|flags| flags[i])
}

fn included_particles(
    selection: ParticleSelection,
    alive_flags: &Option<Vec<bool>>,
    n: usize,
) -> usize {
    if selection.includes_dead() {
        return n;
    }
    alive_flags
        .as_ref()
        .map_or(n, |flags| flags.par_iter().filter(|&&alive| alive).count())
}

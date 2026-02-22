/*!
Generic observer/reducer tools for particle models.
*/

use rayon::prelude::*;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{ATTR_ALIVE, ATTR_M_INV, ATTR_V};

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
    #[inline]
    /// - Purpose: Converts an `AttrsError` into `ObserveError`.
    /// - Parameters:
    ///   - `value` (`AttrsError`): Source error emitted by attribute storage operations.
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
}

#[inline]
/// - Purpose: Reads and validates optional alive-mask data for particle-wise observation.
/// - Parameters:
///   - `objects` (`&PhysObj`): SoA object container queried for optional `alive` attribute.
///   - `n` (`usize`): Expected particle count used to validate the alive-mask length.
fn gather_alive_flags(objects: &PhysObj, n: usize) -> Result<Option<Vec<bool>>, ObserveError> {
    if !objects.core.contains(ATTR_ALIVE) {
        return Ok(None);
    }

    let alive = objects.core.get::<f64>(ATTR_ALIVE)?;
    if alive.dim() != 1 {
        return Err(ObserveError::InvalidAttrShape {
            label: ATTR_ALIVE,
            expected_dim: 1,
            got_dim: alive.dim(),
        });
    }
    if alive.num_vectors() != n {
        return Err(ObserveError::InconsistentParticleCount {
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

#[inline]
/// - Purpose: Returns whether one particle should be included in the current observation pass.
/// - Parameters:
///   - `include_dead` (`bool`): If true, dead particles are still included.
///   - `alive_flags` (`&Option<Vec<bool>>`): Optional alive-mask lookup table by particle index.
///   - `i` (`usize`): Particle index being tested.
fn is_alive(include_dead: bool, alive_flags: &Option<Vec<bool>>, i: usize) -> bool {
    if include_dead {
        return true;
    }
    alive_flags.as_ref().is_none_or(|flags| flags[i])
}

pub trait Observer {
    type Output;

    /// - Purpose: Computes one observable from the current particle state.
    /// - Parameters:
    ///   - `objects` (`&PhysObj`): SoA object container read by this observer.
    fn observe(&self, objects: &PhysObj) -> Result<Self::Output, ObserveError>;
}

pub trait Reducer<T> {
    /// - Purpose: Reduces a list of observed values into one aggregate value.
    /// - Parameters:
    ///   - `values` (`&[T]`): Input batch of values to aggregate.
    fn reduce(&self, values: &[T]) -> T;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MeanReducer;

impl Reducer<f64> for MeanReducer {
    /// - Purpose: Computes arithmetic mean of scalar values.
    /// - Parameters:
    ///   - `values` (`&[f64]`): Scalar input values to average.
    fn reduce(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / (values.len() as f64)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KineticEnergyObserver {
    /// Whether particles marked as dead should still contribute to kinetic energy.
    pub include_dead: bool,
}

impl Default for KineticEnergyObserver {
    /// - Purpose: Constructs a default kinetic-energy observer configuration.
    /// - Parameters:
    fn default() -> Self {
        Self {
            include_dead: false,
        }
    }
}

impl Observer for KineticEnergyObserver {
    type Output = f64;

    /// - Purpose: Computes total kinetic energy `sum_i (0.5 * |v_i|^2 / m_inv_i)` over included particles.
    /// - Parameters:
    ///   - `objects` (`&PhysObj`): SoA object container providing `v`, `m_inv`, and optional `alive`.
    fn observe(&self, objects: &PhysObj) -> Result<Self::Output, ObserveError> {
        let (dim, n) = {
            let v = objects.core.get::<f64>(ATTR_V)?;
            (v.dim(), v.num_vectors())
        };

        let m_inv = objects.core.get::<f64>(ATTR_M_INV)?;
        if m_inv.dim() != 1 {
            return Err(ObserveError::InvalidAttrShape {
                label: ATTR_M_INV,
                expected_dim: 1,
                got_dim: m_inv.dim(),
            });
        }
        if m_inv.num_vectors() != n {
            return Err(ObserveError::InconsistentParticleCount {
                label: ATTR_M_INV,
                expected: n,
                got: m_inv.num_vectors(),
            });
        }

        let alive_flags = gather_alive_flags(objects, n)?;
        let v_data: &[f64] = &objects.core.get::<f64>(ATTR_V)?.as_tensor().data;
        let m_inv_data: &[f64] = &m_inv.as_tensor().data;

        (0..n)
            .into_par_iter()
            .map(|i| -> Result<f64, ObserveError> {
                if !is_alive(self.include_dead, &alive_flags, i) {
                    return Ok(0.0);
                }

                let m_inv_i = m_inv_data[i];
                if !m_inv_i.is_finite() || m_inv_i <= 0.0 {
                    return Err(ObserveError::InvalidState {
                        field: ATTR_M_INV,
                        value: m_inv_i,
                    });
                }

                let row = &v_data[i * dim..(i + 1) * dim];
                let v2 = row.iter().map(|&x| x * x).sum::<f64>();
                Ok(0.5 * v2 / m_inv_i)
            })
            .try_reduce(|| 0.0, |a, b| Ok(a + b))
    }
}

#[inline]
/// - Purpose: Counts particles included by alive-mask policy.
/// - Parameters:
///   - `include_dead` (`bool`): If true, all particles are counted regardless of alive flag.
///   - `alive_flags` (`&Option<Vec<bool>>`): Optional alive-mask lookup table by particle index.
///   - `n` (`usize`): Total particle count.
fn included_particles(include_dead: bool, alive_flags: &Option<Vec<bool>>, n: usize) -> usize {
    if include_dead {
        return n;
    }
    alive_flags
        .as_ref()
        .map_or(n, |flags| flags.par_iter().filter(|&&alive| alive).count())
}

#[derive(Debug, Clone, Copy)]
pub struct TemperatureObserver {
    /// Whether particles marked as dead should still contribute to temperature.
    pub include_dead: bool,
}

impl Default for TemperatureObserver {
    /// - Purpose: Constructs a default temperature observer configuration.
    /// - Parameters:
    fn default() -> Self {
        Self {
            include_dead: false,
        }
    }
}

impl Observer for TemperatureObserver {
    type Output = f64;

    /// - Purpose: Computes kinetic temperature `2*KE/(N_active*dim)` for included particles.
    /// - Parameters:
    ///   - `objects` (`&PhysObj`): SoA object container providing `v` and optional `alive`.
    fn observe(&self, objects: &PhysObj) -> Result<Self::Output, ObserveError> {
        let ke = KineticEnergyObserver {
            include_dead: self.include_dead,
        }
        .observe(objects)?;
        let (dim, n) = {
            let v = objects.core.get::<f64>(ATTR_V)?;
            (v.dim(), v.num_vectors())
        };
        let alive_flags = gather_alive_flags(objects, n)?;
        let count = included_particles(self.include_dead, &alive_flags, n);

        if count == 0 || dim == 0 {
            return Ok(0.0);
        }

        Ok((2.0 * ke) / ((count * dim) as f64))
    }
}

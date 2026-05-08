/*!
Thermostats for canonical massive-particle state.

Purpose:
Thermostats modify particle velocities so a simulated particle system exchanges
energy with an implicit heat bath. This module is particle-specific: it mutates
`ATTR_V`, reads `ATTR_M_INV`, honors `ATTR_ALIVE` through `ParticleSelection`,
and always skips particles marked rigid by `ATTR_RIGID`.

Langevin convention:
`LangevinThermostat` applies the exact Ornstein-Uhlenbeck velocity update

`v_next = exp(-gamma * dt) * v_old + sqrt(tau_target * m_inv * (1 - exp(-2 * gamma * dt))) * z`

where `z` is a standard normal random value generated independently for each
particle and velocity component. The random stream is deterministic for a fixed
`seed`, `step_counter`, particle index, and component traversal order.
*/

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{ATTR_M_INV, ATTR_V, ParticleSelection};
use crate::models::particles::state::{ParticleStateError, gather_inverse_mass, gather_masks};

/// Errors returned by thermostat modules.
#[derive(Debug, Clone, PartialEq)]
pub enum ThermostatError {
    /// Parameter must be finite and within expected range.
    InvalidParam { field: &'static str, value: f64 },
    /// Time-step must be finite and strictly positive.
    InvalidDt { dt: f64 },
    /// Lifted error from `AttrsCore` accessors.
    Attrs(AttrsError),
    /// Invalid per-attribute vector dimension.
    InvalidAttrShape {
        label: &'static str,
        expected_dim: usize,
        got_dim: usize,
    },
    /// Attribute row count mismatch against velocity row count.
    InconsistentParticleCount {
        label: &'static str,
        expected: usize,
        got: usize,
    },
}

impl From<AttrsError> for ThermostatError {
    #[inline]
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
}

impl From<ParticleStateError> for ThermostatError {
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

/// Generic thermostat contract for particle state.
pub trait Thermostat {
    /// Applies one thermostat step with time step `dt`.
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), ThermostatError>;
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Debug, Clone)]
pub struct LangevinThermostat {
    tau_target: f64,
    gamma: f64,
    seed: u64,
    step_counter: u64,
    selection: ParticleSelection,
}

impl LangevinThermostat {
    /// Constructs a Langevin thermostat.
    ///
    /// `tau_target` is the temperature-like energy scale used in the noise
    /// amplitude. `gamma` is the friction rate. Both must be finite and
    /// non-negative. `selection` controls whether dead particles are skipped;
    /// rigid particles are always skipped.
    pub fn new(
        tau_target: f64,
        gamma: f64,
        seed: u64,
        selection: ParticleSelection,
    ) -> Result<Self, ThermostatError> {
        validate_nonnegative("tau_target", tau_target)?;
        validate_nonnegative("gamma", gamma)?;

        Ok(Self {
            tau_target,
            gamma,
            seed,
            step_counter: 0,
            selection,
        })
    }

    #[inline]
    pub fn tau_target(&self) -> f64 {
        self.tau_target
    }

    #[inline]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[inline]
    pub fn step_counter(&self) -> u64 {
        self.step_counter
    }

    #[inline]
    pub fn selection(&self) -> ParticleSelection {
        self.selection
    }
}

impl Thermostat for LangevinThermostat {
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), ThermostatError> {
        validate_dt(dt)?;

        let (dim, n) = {
            let v = objects.core.get::<f64>(ATTR_V)?;
            (v.dim(), v.num_vectors())
        };

        let m_inv_values = gather_inverse_mass(objects, n)?;
        let masks = gather_masks(objects, n, self.selection)?;

        let c = (-self.gamma * dt).exp();
        let one_minus_c2 = (1.0 - c * c).max(0.0);
        let step = self.step_counter;
        let seed = self.seed;
        let tau_target = self.tau_target;

        let v = objects.core.get_mut::<f64>(ATTR_V)?;
        v.as_tensor_mut()
            .data
            .par_chunks_mut(dim)
            .enumerate()
            .try_for_each(|(i, row)| -> Result<(), ThermostatError> {
                if masks.should_skip(i) {
                    return Ok(());
                }

                let m_inv = m_inv_values[i];
                if !m_inv.is_finite() || m_inv <= 0.0 {
                    return Err(ThermostatError::InvalidParam {
                        field: ATTR_M_INV,
                        value: m_inv,
                    });
                }

                let sigma = (tau_target * m_inv * one_minus_c2).sqrt();
                if !sigma.is_finite() {
                    return Err(ThermostatError::InvalidParam {
                        field: "sigma",
                        value: sigma,
                    });
                }

                let row_seed = splitmix64(seed ^ step ^ ((i as u64) << 1));
                let mut rng = SmallRng::seed_from_u64(row_seed);
                for vd in row.iter_mut() {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    *vd = c * *vd + sigma * z;
                }

                Ok(())
            })?;

        self.step_counter = self.step_counter.wrapping_add(1);
        Ok(())
    }
}

#[inline]
fn validate_nonnegative(field: &'static str, value: f64) -> Result<(), ThermostatError> {
    if !value.is_finite() || value < 0.0 {
        return Err(ThermostatError::InvalidParam { field, value });
    }
    Ok(())
}

#[inline]
fn validate_dt(dt: f64) -> Result<(), ThermostatError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(ThermostatError::InvalidDt { dt });
    }
    Ok(())
}

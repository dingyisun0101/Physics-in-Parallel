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

use core::fmt;

use rayon::prelude::*;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{ATTR_M_INV, ATTR_V, ParticleSelection};
use crate::models::particles::state::{ParticleStateError, gather_inverse_mass, gather_masks};
use crate::rng::{ResolvedRng, RngError};
use crate::space::discrete::square_lattice::random::IndexedRng;

const DOMAIN_LANGEVIN_NORMAL: u64 = 0xc7c7_d252_9b53_6071;

/// Errors returned by thermostat modules.
#[derive(Debug, Clone, PartialEq)]
pub enum ThermostatError {
    /// Parameter must be finite and within expected range.
    InvalidParam {
        field: &'static str,
        value: f64,
    },
    /// Time-step must be finite and strictly positive.
    InvalidDt {
        dt: f64,
    },
    State(ParticleStateError),
    Rng(RngError),
}

impl From<AttrsError> for ThermostatError {
    #[inline]
    fn from(value: AttrsError) -> Self {
        Self::State(value.into())
    }
}

impl From<ParticleStateError> for ThermostatError {
    fn from(value: ParticleStateError) -> Self {
        Self::State(value)
    }
}

impl fmt::Display for ThermostatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParam { field, value } => write!(
                f,
                "thermostat parameter `{field}` must be finite and non-negative; got {value}"
            ),
            Self::InvalidDt { dt } => {
                write!(
                    f,
                    "thermostat time step must be finite and positive; got {dt}"
                )
            }
            Self::State(error) => write!(f, "invalid particle state: {error}"),
            Self::Rng(error) => write!(f, "thermostat RNG error: {error}"),
        }
    }
}

impl std::error::Error for ThermostatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(error) => Some(error),
            Self::Rng(error) => Some(error),
            _ => None,
        }
    }
}

/// Generic thermostat contract for particle state.
pub trait Thermostat: Send {
    /// Applies one thermostat step with time step `dt`.
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), ThermostatError>;
}

#[derive(Debug, Clone)]
pub struct LangevinThermostat {
    tau_target: f64,
    gamma: f64,
    rng: IndexedRng,
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
        rng: ResolvedRng,
        selection: ParticleSelection,
    ) -> Result<Self, ThermostatError> {
        validate_nonnegative("tau_target", tau_target)?;
        validate_nonnegative("gamma", gamma)?;
        let rng = IndexedRng::new(rng).map_err(ThermostatError::Rng)?;

        Ok(Self {
            tau_target,
            gamma,
            rng,
            step_counter: 0,
            selection,
        })
    }

    /// Restores a Langevin thermostat from resolved deterministic state.
    ///
    /// Unlike [`Self::new`], this constructor never fills missing RNG values
    /// from defaults or host entropy. `rng` must contain both a seed and the
    /// indexed SplitMix64 method previously returned by [`Self::resolved_rng`].
    pub fn from_state(
        tau_target: f64,
        gamma: f64,
        rng: ResolvedRng,
        step_counter: u64,
        selection: ParticleSelection,
    ) -> Result<Self, ThermostatError> {
        validate_nonnegative("tau_target", tau_target)?;
        validate_nonnegative("gamma", gamma)?;
        let rng = IndexedRng::new(rng).map_err(ThermostatError::Rng)?;

        Ok(Self {
            tau_target,
            gamma,
            rng,
            step_counter,
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
    pub fn resolved_rng(&self) -> ResolvedRng {
        self.rng.resolved_rng()
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
        let rng = self.rng;
        let tau_target = self.tau_target;

        let v = objects.core.get_mut::<f64>(ATTR_V)?;
        v.edit_values(|values| {
            values.par_chunks_mut(dim).enumerate().try_for_each(
                |(i, row)| -> Result<(), ThermostatError> {
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

                    for (component, vd) in row.iter_mut().enumerate() {
                        let z = rng.standard_normal(
                            step,
                            DOMAIN_LANGEVIN_NORMAL,
                            i as u64,
                            component as u64,
                        );
                        *vd = c * *vd + sigma * z;
                    }

                    Ok(())
                },
            )
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

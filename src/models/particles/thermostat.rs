/*!
Thermostat interfaces and implementations for massive-particle models.
*/

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{ATTR_ALIVE, ATTR_M_INV, ATTR_RIGID, ATTR_V};

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

/// Generic thermostat contract.
pub trait Thermostat {
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
    include_dead: bool,
}

impl LangevinThermostat {
    pub fn new(
        tau_target: f64,
        gamma: f64,
        seed: u64,
        include_dead: bool,
    ) -> Result<Self, ThermostatError> {
        if !tau_target.is_finite() || tau_target < 0.0 {
            return Err(ThermostatError::InvalidParam {
                field: "tau_target",
                value: tau_target,
            });
        }
        if !gamma.is_finite() || gamma < 0.0 {
            return Err(ThermostatError::InvalidParam {
                field: "gamma",
                value: gamma,
            });
        }

        Ok(Self {
            tau_target,
            gamma,
            seed,
            step_counter: 0,
            include_dead,
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
    pub fn step_counter(&self) -> u64 {
        self.step_counter
    }
}

impl Thermostat for LangevinThermostat {
    fn apply(&mut self, objects: &mut PhysObj, dt: f64) -> Result<(), ThermostatError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ThermostatError::InvalidDt { dt });
        }

        let (dim, n) = {
            let v = objects.core.get::<f64>(ATTR_V)?;
            (v.dim(), v.num_vectors())
        };

        let m_inv_values: Vec<f64> = {
            let m_inv = objects.core.get::<f64>(ATTR_M_INV)?;
            if m_inv.dim() != 1 {
                return Err(ThermostatError::InvalidAttrShape {
                    label: ATTR_M_INV,
                    expected_dim: 1,
                    got_dim: m_inv.dim(),
                });
            }
            if m_inv.num_vectors() != n {
                return Err(ThermostatError::InconsistentParticleCount {
                    label: ATTR_M_INV,
                    expected: n,
                    got: m_inv.num_vectors(),
                });
            }

            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(m_inv.get(i as isize, 0));
            }
            out
        };

        let alive_flags: Option<Vec<bool>> = if self.include_dead || !objects.core.contains(ATTR_ALIVE) {
            None
        } else {
            let alive = objects.core.get::<f64>(ATTR_ALIVE)?;
            if alive.dim() != 1 {
                return Err(ThermostatError::InvalidAttrShape {
                    label: ATTR_ALIVE,
                    expected_dim: 1,
                    got_dim: alive.dim(),
                });
            }
            if alive.num_vectors() != n {
                return Err(ThermostatError::InconsistentParticleCount {
                    label: ATTR_ALIVE,
                    expected: n,
                    got: alive.num_vectors(),
                });
            }

            let mut flags = Vec::with_capacity(n);
            for i in 0..n {
                flags.push(alive.get(i as isize, 0) > 0.0);
            }
            Some(flags)
        };

        let rigid_flags: Option<Vec<bool>> = if !objects.core.contains(ATTR_RIGID) {
            None
        } else {
            let rigid = objects.core.get::<f64>(ATTR_RIGID)?;
            if rigid.dim() != 1 {
                return Err(ThermostatError::InvalidAttrShape {
                    label: ATTR_RIGID,
                    expected_dim: 1,
                    got_dim: rigid.dim(),
                });
            }
            if rigid.num_vectors() != n {
                return Err(ThermostatError::InconsistentParticleCount {
                    label: ATTR_RIGID,
                    expected: n,
                    got: rigid.num_vectors(),
                });
            }

            let mut flags = Vec::with_capacity(n);
            for i in 0..n {
                flags.push(rigid.get(i as isize, 0) > 0.0);
            }
            Some(flags)
        };

        let c = (-self.gamma * dt).exp();
        let one_minus_c2 = (1.0 - c * c).max(0.0);

        let v = objects.core.get_mut::<f64>(ATTR_V)?;
        for i in 0..n {
            if let Some(flags) = &alive_flags {
                if !flags[i] {
                    continue;
                }
            }
            if let Some(flags) = &rigid_flags {
                if flags[i] {
                    continue;
                }
            }

            let m_inv = m_inv_values[i];
            if !m_inv.is_finite() || m_inv <= 0.0 {
                return Err(ThermostatError::InvalidParam {
                    field: "m_inv",
                    value: m_inv,
                });
            }

            let sigma = (self.tau_target * m_inv * one_minus_c2).sqrt();
            if !sigma.is_finite() {
                return Err(ThermostatError::InvalidParam {
                    field: "sigma",
                    value: sigma,
                });
            }

            let seed = splitmix64(self.seed ^ self.step_counter ^ ((i as u64) << 1));
            let mut rng = SmallRng::seed_from_u64(seed);
            let row = v.get_vector_mut(i as isize);

            for vd in row.iter_mut().take(dim) {
                let z: f64 = StandardNormal.sample(&mut rng);
                *vd = c * *vd + sigma * z;
            }
        }

        self.step_counter = self.step_counter.wrapping_add(1);
        Ok(())
    }
}

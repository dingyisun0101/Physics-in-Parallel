/*!
Langevin thermostat for `engines::soa`.

Update rule (per component):
`v <- c * v + sigma * z`
where
- `c = exp(-gamma * dt)`
- `sigma = sqrt(tau_target * inv_mass * (1 - c^2))`
- `z ~ N(0, 1)`
*/

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};

use crate::engines::soa::phys_obj::PhysObj;

use super::{Thermostat, ThermostatError};

#[inline]
/// Annotation:
/// - Purpose: Executes `splitmix64` logic for this module.
/// - Parameters:
///   - `mut x` (`u64`): Parameter of type `u64` used by `splitmix64`.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Langevin thermostat state.
#[derive(Debug, Clone)]
pub struct LangevinThermostat {
    tau_target: f64,
    gamma: f64,
    seed: u64,
    step_counter: u64,
    include_dead: bool,
}

impl LangevinThermostat {
    /// Build with deterministic RNG seed.
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `tau_target` (`f64`): Temperature-like scaling parameter.
    ///   - `gamma` (`f64`): Damping/decay parameter.
    ///   - `seed` (`u64`): Random seed controlling deterministic sampling.
    ///   - `include_dead` (`bool`): Parameter of type `bool` used by `new`.
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
    /// Annotation:
    /// - Purpose: Executes `tau_target` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn tau_target(&self) -> f64 {
        self.tau_target
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `gamma` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `step_counter` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn step_counter(&self) -> u64 {
        self.step_counter
    }
}

impl<const D: usize> Thermostat<D> for LangevinThermostat {
    /// Annotation:
    /// - Purpose: Executes `apply` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj<D>`): Object-state container used by this operation.
    ///   - `dt` (`f64`): Time-step used for one simulation/integration update.
    fn apply(&mut self, objects: &mut PhysObj<D>, dt: f64) -> Result<(), ThermostatError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ThermostatError::InvalidDt { dt });
        }

        let c = (-self.gamma * dt).exp();
        let one_minus_c2 = (1.0 - c * c).max(0.0);
        let n = objects.len();

        for i in 0..n {
            if !self.include_dead && !objects.is_alive(i)? {
                continue;
            }

            let inv_mass = objects.inv_mass_of(i)?;
            if !inv_mass.is_finite() || inv_mass <= 0.0 {
                return Err(ThermostatError::InvalidParam {
                    field: "inv_mass",
                    value: inv_mass,
                });
            }

            let sigma = (self.tau_target * inv_mass * one_minus_c2).sqrt();
            if !sigma.is_finite() {
                return Err(ThermostatError::InvalidParam {
                    field: "sigma",
                    value: sigma,
                });
            }

            let seed = splitmix64(self.seed ^ self.step_counter ^ ((i as u64) << 1));
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut v = objects.vel_of(i)?.to_vec();

            for vd in v.iter_mut().take(D) {
                let z: f64 = StandardNormal.sample(&mut rng);
                *vd = c * *vd + sigma * z;
            }
            objects.set_vel(i, &v)?;
        }

        self.step_counter = self.step_counter.wrapping_add(1);
        Ok(())
    }
}

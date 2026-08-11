/*!
User-facing prelude for model types.

Import with:
`use physics_in_parallel::models::prelude::*;`

Purpose:
This prelude exposes the stable model API that most users need directly:
validated law payloads, canonical particle attribute helpers, particle state
construction, particle observers, integrators, thermostats, boundary adapters,
and particle interaction wrappers.
*/

pub use crate::rng::{IndexedRng, RngConfig, RngConfigError, RngMethod};

pub use crate::models::laws::{
    PowerLawDecay, PowerLawError, PowerLawRange, Spring, SpringCutoff, SpringLawError,
};
pub use crate::models::particles::attrs::{
    ALIVE_FALSE, ALIVE_TRUE, ATTR_A, ATTR_ALIVE, ATTR_M, ATTR_M_INV, ATTR_R, ATTR_RIGID, ATTR_V,
    ParticleSelection, RIGID_FALSE, RIGID_TRUE, alive_value, is_alive, is_alive_value, is_rigid,
    is_rigid_value, rigid_value, set_alive, set_rigid,
};
pub use crate::models::particles::boundary::{ParticleBoundary, ParticleBoundaryError};
pub use crate::models::particles::create_state::{
    MassiveParticlesError, VelocitySamplingMethod, create_template, randomize_r, randomize_v,
};
pub use crate::models::particles::integrator::{
    ExplicitEuler, Integrator, IntegratorError, SemiImplicitEuler,
};
pub use crate::models::particles::interactions::{
    ParticleNeighborList, ParticleNeighborListError, PowerLawNetwork, PowerLawNetworkError,
    SpringNetwork, SpringNetworkError,
};
pub use crate::models::particles::observe::{
    KineticEnergyObserver, ObserveError, Observer, TemperatureObserver,
};
pub use crate::models::particles::thermostat::{LangevinThermostat, Thermostat, ThermostatError};

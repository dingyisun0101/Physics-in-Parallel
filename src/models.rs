/*!
Domain-level model modules.

Purpose:
`models` contains concrete reusable physical model components. These modules
should build on `math`, `space`, and `engines` rather than inventing their own
storage or geometry rules. Current ready pieces cover validated law payloads
and canonical massive-particle state.
*/

pub mod laws;
pub mod particles;

pub use crate::engines::soa::phys_obj::ParticleStateError;
pub use laws::{PowerLawDecay, PowerLawError, PowerLawRange, Spring, SpringCutoff, SpringLawError};
pub use particles::attrs::{
    ATTR_A, ATTR_ALIVE, ATTR_M, ATTR_M_INV, ATTR_R, ATTR_RIGID, ATTR_V, ParticleSelection,
    is_alive, is_rigid, set_alive, set_rigid,
};
pub use particles::boundary::{ParticleBoundary, ParticleBoundaryError};
pub use particles::create_state::{
    MassiveParticlesError, VelocitySamplingMethod, create_template, randomize_r, randomize_v,
};
pub use particles::integrator::{ExplicitEuler, Integrator, IntegratorError, SemiImplicitEuler};
pub use particles::interactions::{
    ParticleNeighborList, ParticleNeighborListError, PowerLawNetwork, PowerLawNetworkError,
    SpringNetwork, SpringNetworkError,
};
pub use particles::observe::{KineticEnergyObserver, ObserveError, Observer, TemperatureObserver};
pub use particles::thermostat::{LangevinThermostat, Thermostat, ThermostatError};

pub use crate::engines::soa::phys_obj::PhysObj;

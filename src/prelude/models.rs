//! Ready-to-use physical laws and canonical particle-model API.

pub use crate::models::{
    ATTR_A, ATTR_ALIVE, ATTR_M, ATTR_M_INV, ATTR_R, ATTR_RIGID, ATTR_V, AttrsError, ExplicitEuler,
    Integrator, IntegratorError, KineticEnergyObserver, LangevinThermostat, MassiveParticlesError,
    ObserveError, Observer, ParticleBoundary, ParticleBoundaryError, ParticleNeighborList,
    ParticleNeighborListError, ParticleSelection, ParticleStateError, PhysObj, PowerLawDecay,
    PowerLawError, PowerLawNetwork, PowerLawNetworkError, PowerLawRange, SemiImplicitEuler, Spring,
    SpringCutoff, SpringLawError, SpringNetwork, SpringNetworkError, TemperatureObserver,
    Thermostat, ThermostatError, VelocitySamplingMethod, create_template, is_alive, is_rigid,
    randomize_r, randomize_v, set_alive, set_rigid,
};

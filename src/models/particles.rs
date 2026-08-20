/*!
Particle-model modules.

Purpose:
This module contains reusable model pieces for massive-particle simulations:
canonical attribute labels, state construction, boundary handling, time
integration, thermostatting, observations, and pair-interaction helpers.
*/

pub(crate) mod attrs;
pub(crate) mod boundary;
pub(crate) mod create_state;
pub(crate) mod integrator;
pub(crate) mod interactions;
pub(crate) mod observe;
pub(crate) mod state;
pub(crate) mod thermostat;

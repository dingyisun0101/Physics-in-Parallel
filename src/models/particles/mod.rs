/*!
Particle-model modules.

Purpose:
This module contains reusable model pieces for massive-particle simulations:
canonical attribute labels, state construction, boundary handling, time
integration, thermostatting, observations, and pair-interaction helpers.
*/

pub mod attrs;
pub mod boundary;
pub mod create_state;
pub mod integrator;
pub mod interactions;
pub mod observe;
pub(crate) mod state;
pub mod thermostat;

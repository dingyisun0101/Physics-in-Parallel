/*!
Reusable physical interaction-law payloads.

Purpose:
This module stores small, validated parameter objects for physical laws. A law
payload does not know how particles, lattice sites, or fields are stored. Model
adapters decide how to apply these parameters to concrete state.
*/

pub mod power_law;
pub mod spring;

pub use power_law::{PowerLawDecay, PowerLawError, PowerLawRange};
pub use spring::{Spring, SpringCutoff, SpringLawError};

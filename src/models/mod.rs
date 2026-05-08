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
pub mod prelude;

pub use laws::*;
pub use particles::{attrs, boundary, create_state, integrator, interactions, observe, thermostat};

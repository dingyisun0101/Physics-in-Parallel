/*!
Simulation engine infrastructure.

Purpose:
    `engines` contains model-agnostic runtime containers and interaction
    backends. The current ready backend is structure-of-arrays (`soa`), which
    stores object attributes as typed vector-list columns and interaction
    payloads behind stable topology slots.
*/

pub mod prelude;
pub mod soa;
pub use soa::*;

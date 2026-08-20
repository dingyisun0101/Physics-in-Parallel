/*!
Canonical attribute labels and scalar flag helpers for massive-particle models.

Purpose:
This module defines the vocabulary used by particle modules. It does not gather
whole columns or validate multi-particle state; `particles::state` handles that
shared interpretation. Here we keep only labels and small single-particle flag
helpers that are convenient for end users.

Canonical attribute shapes:
- `ATTR_R`: position vectors, shape `[num_particles, dim]`, scalar type `f64`.
- `ATTR_V`: velocity vectors, shape `[num_particles, dim]`, scalar type `f64`.
- `ATTR_A`: acceleration vectors, shape `[num_particles, dim]`, scalar type `f64`.
- `ATTR_M`: mass scalars, shape `[num_particles, 1]`, scalar type `f64`.
- `ATTR_M_INV`: inverse-mass scalars, shape `[num_particles, 1]`, scalar type `f64`.
- `ATTR_ALIVE`: alive/dead flags, shape `[num_particles, 1]`, scalar type `u8`.
- `ATTR_RIGID`: rigid/fixed flags, shape `[num_particles, 1]`, scalar type `u8`.

Flag conventions:
`alive` uses compact `u8` storage, but callers should normally use bool-facing
helpers. `rigid` uses the same compact `u8` convention because it is also a
logical mask rather than a floating-point physical field.
*/

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};

/// Canonical position attribute label.
pub const ATTR_R: &str = "r";
/// Canonical velocity attribute label.
pub const ATTR_V: &str = "v";
/// Canonical acceleration attribute label.
pub const ATTR_A: &str = "a";
/// Canonical mass attribute label.
pub const ATTR_M: &str = "m";
/// Canonical inverse-mass attribute label.
pub const ATTR_M_INV: &str = "m_inv";
/// Canonical alive-mask scalar label.
pub const ATTR_ALIVE: &str = "alive";
/// Canonical rigid-mask scalar label (`>0` means this particle is rigid/fixed).
pub const ATTR_RIGID: &str = "rigid";

/// Stored value for a dead/inactive particle.
pub(crate) const ALIVE_FALSE: u8 = 0;
/// Stored value for an alive/active particle.
pub(crate) const ALIVE_TRUE: u8 = 1;
/// Stored value for a non-rigid/free particle.
pub(crate) const RIGID_FALSE: u8 = 0;
/// Stored value for a rigid/fixed particle.
pub(crate) const RIGID_TRUE: u8 = 1;

/// Converts a Rust boolean into the canonical stored alive-mask value.
pub(crate) fn alive_value(alive: bool) -> u8 {
    if alive { ALIVE_TRUE } else { ALIVE_FALSE }
}

/// Interprets a canonical alive-mask value.
pub(crate) fn is_alive_value(value: u8) -> bool {
    value != ALIVE_FALSE
}

/// Converts a Rust boolean into the canonical stored rigid-mask value.
pub(crate) fn rigid_value(rigid: bool) -> u8 {
    if rigid { RIGID_TRUE } else { RIGID_FALSE }
}

/// Interprets a canonical rigid-mask value.
pub(crate) fn is_rigid_value(value: u8) -> bool {
    value != RIGID_FALSE
}

/// Sets one particle's alive/dead state.
pub fn set_alive(objects: &mut PhysObj, i: usize, alive: bool) -> Result<(), AttrsError> {
    objects
        .core
        .set_vector_of::<u8>(ATTR_ALIVE, i, &[alive_value(alive)])
}

/// Returns one particle's alive/dead state.
pub fn is_alive(objects: &PhysObj, i: usize) -> Result<bool, AttrsError> {
    Ok(is_alive_value(
        objects.core.vector_of::<u8>(ATTR_ALIVE, i)?[0],
    ))
}

/// Sets one particle's rigid/free state.
pub fn set_rigid(objects: &mut PhysObj, i: usize, rigid: bool) -> Result<(), AttrsError> {
    objects
        .core
        .set_vector_of::<u8>(ATTR_RIGID, i, &[rigid_value(rigid)])
}

/// Returns one particle's rigid/free state.
pub fn is_rigid(objects: &PhysObj, i: usize) -> Result<bool, AttrsError> {
    Ok(is_rigid_value(
        objects.core.vector_of::<u8>(ATTR_RIGID, i)?[0],
    ))
}

/// Particle subset used by operations that can honor `ATTR_ALIVE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleSelection {
    /// Normal physical path: only particles with `alive > 0` are included.
    AliveOnly,
    /// Debugging/inspection path: all allocated particle slots are included.
    ///
    /// This intentionally ignores `ATTR_ALIVE`. Use it for diagnostics,
    /// checkpoint inspection, or slot-recycling workflows, not for ordinary
    /// physical evolution.
    All,
}

impl ParticleSelection {
    /// Returns true when dead particles should be included.
    pub fn includes_dead(self) -> bool {
        matches!(self, Self::All)
    }
}

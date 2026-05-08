/*!
Shared attribute labels and small access helpers for massive-particle models.
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
pub const ALIVE_FALSE: u8 = 0;
/// Stored value for an alive/active particle.
pub const ALIVE_TRUE: u8 = 1;

/// Converts a Rust boolean into the canonical stored alive-mask value.
pub fn alive_value(alive: bool) -> u8 {
    if alive { ALIVE_TRUE } else { ALIVE_FALSE }
}

/// Interprets a canonical alive-mask value.
pub fn is_alive_value(value: u8) -> bool {
    value != ALIVE_FALSE
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

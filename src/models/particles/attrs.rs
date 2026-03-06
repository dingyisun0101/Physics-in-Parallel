/*!
Shared attribute labels for massive-particle model modules.
*/

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

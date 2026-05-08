/*!
Crate-level IO convenience exports.

Purpose:
This module currently re-exports math IO utilities for callers who want a short
`physics_in_parallel::io` path. Space-specific IO remains under `space::io`
because it carries geometry-specific contracts such as square-lattice boundary
tags.
*/

pub use crate::math::io::*;

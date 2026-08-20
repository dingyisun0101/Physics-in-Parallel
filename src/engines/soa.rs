/*!
Structure-of-arrays simulation backend.

Purpose:
    This module stores object state by attribute column instead of by object
    struct. It is the right layout when model code repeatedly updates the same
    attribute for many objects, because each attribute is contiguous and can be
    processed in parallel.
*/

pub(crate) mod interaction;
pub(crate) mod neighbor_list;
pub(crate) mod phys_obj;

// Canonical SoA exports.
pub(crate) use interaction::{Interaction, InteractionError, InteractionId};
pub use neighbor_list::{NeighborList, NeighborListError};

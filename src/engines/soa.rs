/*!
Structure-of-arrays simulation backend.

Purpose:
    This module stores object state by attribute column instead of by object
    struct. It is the right layout when model code repeatedly updates the same
    attribute for many objects, because each attribute is contiguous and can be
    processed in parallel.
*/

pub mod interaction;
pub mod neighbor_list;
pub mod phys_obj;

// Canonical SoA exports.
pub use interaction::{
    Interaction, InteractionError, InteractionId, InteractionNodes, InteractionOrder,
    InteractionTopology, ObjId,
};
pub use neighbor_list::{NeighborList, NeighborListError};
pub use phys_obj::{AttrId, AttrsCore, AttrsError, AttrsMeta, PhysObj};

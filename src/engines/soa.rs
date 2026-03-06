/*!
Structure-of-arrays engine implementation.
*/

pub mod interaction;
pub mod neighbor_list;
pub mod phys_obj;

// Canonical SoA exports.
pub use interaction::{EdgeId, Interaction, InteractionError, Topology};
pub use neighbor_list::{NeighborList, NeighborListError};
pub use phys_obj::{AttrId, AttrsCore, AttrsError, AttrsMeta, PhysObj};

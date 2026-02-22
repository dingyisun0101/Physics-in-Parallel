/*!
Structure-of-arrays engine implementation.
*/

pub mod interaction;
pub mod phys_obj;

// Canonical SoA exports.
pub use interaction::{EdgeId, Interaction, InteractionError, Topology};
pub use phys_obj::{AttrId, AttrsCore, AttrsError, AttrsMeta, PhysObj};

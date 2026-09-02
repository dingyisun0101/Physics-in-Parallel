//! Serde, JSON-schema, and textual interop for PiP math types.
//!
//! Dense and sparse tensors, matrices, and vector lists implement Serde at
//! their concrete type boundaries. Applications should normally call a Serde
//! serializer or deserializer directly; the payload traits in [`json`] expose
//! owned schema objects only when explicit schema manipulation is useful.
//!
//! Persistence policy is intentionally outside this module. PiP defines and
//! validates scientific payload representations but does not manage recording
//! cadence, chunks, queues, checkpoints, or output-directory organization.

pub(crate) mod json;
pub(crate) mod matrix;
pub(crate) mod tensor;
pub(crate) mod vector_list;

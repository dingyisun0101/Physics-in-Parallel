//! Serde, JSON-schema, ndarray, and textual interop for PiP math types.
//!
//! Dense and sparse tensors, matrices, and vector lists implement Serde at
//! their concrete type boundaries. Applications should normally call a Serde
//! serializer or deserializer directly; the payload traits in [`json`] expose
//! owned schema objects only when explicit schema manipulation is useful.
//!
//! Persistence policy is intentionally outside this module. PiP defines and
//! validates scientific payload representations but does not manage recording
//! cadence, chunks, queues, checkpoints, or output-directory organization.

pub mod json;
pub mod matrix;
pub mod ndarray;
pub mod string;
pub mod tensor;
pub mod vector_list;

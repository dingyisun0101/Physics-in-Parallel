/*!
N-body interaction storage for the structure-of-arrays engine.

Purpose:
This module represents relationships between simulation objects. A relationship
can be pairwise, such as a spring between two particles, or higher order, such
as a three-body angle term. The participating object indices are stored once in
an `InteractionTopology`; the numerical or model-specific payload is stored in
`Interaction<T>` at the matching interaction id.

Design:
`InteractionOrder::Ordered` treats `[i, j]` and `[j, i]` as different
interactions. Use it for interactions where the order of participants carries
meaning.

`InteractionOrder::Unordered` canonicalizes the participant list by sorting it.
Use it for symmetric interactions where the same physical term is obtained
regardless of the order in which objects are supplied.

Topology remains mutable because downstream models may grow object counts,
switch ordering semantics, or prune invalid interactions during a simulation.
When payloads are present, prefer the mutation methods on `Interaction<T>` so
the topology and payload storage remain synchronized.
*/

use ahash::AHashMap;
use core::fmt;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::phys_obj::AttrsError;

/// Stable object index in `PhysObj` columns.
pub type ObjId = usize;

/// Stable id for one active or reusable interaction slot.
pub type InteractionId = usize;

/// Whether participant order changes the identity of an interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionOrder {
    /// Preserve caller-provided node order exactly.
    Ordered,
    /// Sort node order before lookup/insert/remove.
    Unordered,
}

/// Object indices participating in one interaction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionNodes {
    /// Object indices identifying one interaction term; arity is `nodes.len()`.
    pub nodes: Box<[ObjId]>,
}

impl std::borrow::Borrow<[ObjId]> for InteractionNodes {
    fn borrow(&self) -> &[ObjId] {
        &self.nodes
    }
}

impl InteractionNodes {
    /// Builds an owned interaction-node list from object indices.
    pub fn from_slice(nodes: &[ObjId]) -> Self {
        Self {
            nodes: nodes.into(),
        }
    }

    /// Builds an owned pair node list.
    pub fn from_pair(i: ObjId, j: ObjId) -> Self {
        Self {
            nodes: Box::new([i, j]),
        }
    }

    /// Returns the number of participating objects.
    pub fn arity(&self) -> usize {
        self.nodes.len()
    }
}

/// Errors returned by interaction backend operations.
#[derive(Debug, Clone, PartialEq)]
pub enum InteractionError {
    /// Interaction node list is empty.
    EmptyNodes,
    /// Object index is outside topology bounds.
    InvalidObjId {
        /// Invalid object index value.
        obj: ObjId,
        /// Current exclusive upper bound (`0..n_objects`).
        n_objects: usize,
    },
    /// Interaction id does not exist or is inactive.
    InvalidInteractionId {
        /// Requested interaction id.
        id: InteractionId,
        /// Total slot capacity at time of check.
        n_slots: usize,
    },
    /// Changing from ordered to unordered interactions would merge two entries.
    OrderChangeCollision {
        /// Canonical node list shared by at least two active interactions.
        nodes: Box<[ObjId]>,
        /// Existing interaction id already using the canonical node list.
        existing: InteractionId,
        /// New interaction id that would collide with the existing one.
        incoming: InteractionId,
    },
    /// Requested object-count shrink would invalidate an existing interaction.
    ObjectCountWouldInvalidate {
        /// Requested new object count.
        n_objects: usize,
        /// First active interaction id that would become invalid.
        id: InteractionId,
        /// First offending object index in that interaction.
        obj: ObjId,
    },
    /// Restored topology contains the same interaction more than once.
    DuplicateNodes {
        /// Canonical participant list shared by both ids.
        nodes: Box<[ObjId]>,
        /// First interaction id using the participant list.
        existing: InteractionId,
        /// Later interaction id using the same participant list.
        incoming: InteractionId,
    },
    /// Wrapped attribute/core error from `PhysObj` operations.
    Attrs(
        /// Lower-level attribute/core error details.
        AttrsError,
    ),
}

impl From<AttrsError> for InteractionError {
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
}

impl fmt::Display for InteractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyNodes => write!(f, "interaction must contain at least one object"),
            Self::InvalidObjId { obj, n_objects } => write!(
                f,
                "interaction object index {obj} is out of bounds for {n_objects} objects"
            ),
            Self::InvalidInteractionId { id, n_slots } => write!(
                f,
                "interaction id {id} is inactive or out of bounds for {n_slots} slots"
            ),
            Self::OrderChangeCollision {
                nodes,
                existing,
                incoming,
            } => write!(
                f,
                "changing interaction order would merge ids {existing} and {incoming} at nodes {nodes:?}"
            ),
            Self::ObjectCountWouldInvalidate { n_objects, id, obj } => write!(
                f,
                "reducing to {n_objects} objects would invalidate interaction {id} at object {obj}"
            ),
            Self::DuplicateNodes {
                nodes,
                existing,
                incoming,
            } => write!(
                f,
                "interaction ids {existing} and {incoming} contain duplicate nodes {nodes:?}"
            ),
            Self::Attrs(error) => write!(f, "interaction attribute error: {error}"),
        }
    }
}

impl std::error::Error for InteractionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Attrs(error) => Some(error),
            _ => None,
        }
    }
}

/// Mutable n-body topology mapping participant lists to stable interaction ids.
#[derive(Debug, Clone)]
pub struct InteractionTopology {
    /// Exclusive upper bound for valid object indices (`0..n_objects`).
    n_objects: usize,
    /// Whether object order is part of interaction identity.
    order: InteractionOrder,
    /// Forward mapping from interaction nodes to stable id.
    id_of_nodes: AHashMap<InteractionNodes, InteractionId>,
    /// Reverse mapping from id to nodes; `None` means the slot is currently free.
    nodes_of_id: Vec<Option<InteractionNodes>>,
    /// Reusable free ids used for O(1)-average insert/remove churn.
    free_ids: Vec<InteractionId>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InteractionTopologySerde {
    n_objects: usize,
    order: InteractionOrder,
    slots: Vec<Option<InteractionNodes>>,
}

impl Serialize for InteractionTopology {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        InteractionTopologySerde {
            n_objects: self.n_objects,
            order: self.order,
            slots: self.nodes_of_id.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for InteractionTopology {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let state = InteractionTopologySerde::deserialize(deserializer)?;
        Self::from_serialized_slots(state.n_objects, state.order, state.slots)
            .map_err(serde::de::Error::custom)
    }
}

impl InteractionTopology {
    fn from_serialized_slots(
        n_objects: usize,
        order: InteractionOrder,
        mut nodes_of_id: Vec<Option<InteractionNodes>>,
    ) -> Result<Self, InteractionError> {
        let mut id_of_nodes = AHashMap::with_capacity(nodes_of_id.len());

        for (id, maybe_nodes) in nodes_of_id.iter_mut().enumerate() {
            let Some(nodes) = maybe_nodes else {
                continue;
            };
            if nodes.nodes.is_empty() {
                return Err(InteractionError::EmptyNodes);
            }
            for &obj in nodes.nodes.iter() {
                if obj >= n_objects {
                    return Err(InteractionError::InvalidObjId { obj, n_objects });
                }
            }

            let canonical = InteractionNodes::from_slice(&canonicalize_nodes(&nodes.nodes, order));
            if let Some(existing) = id_of_nodes.insert(canonical.clone(), id) {
                return Err(InteractionError::DuplicateNodes {
                    nodes: canonical.nodes,
                    existing,
                    incoming: id,
                });
            }
            *nodes = canonical;
        }

        let free_ids = nodes_of_id
            .iter()
            .enumerate()
            .rev()
            .filter_map(|(id, nodes)| nodes.is_none().then_some(id))
            .collect();

        Ok(Self {
            n_objects,
            order,
            id_of_nodes,
            nodes_of_id,
            free_ids,
        })
    }

    /// Constructs an empty unordered topology.
    pub fn new(n_objects: usize) -> Self {
        Self::with_order(n_objects, InteractionOrder::Unordered)
    }

    /// Constructs an empty topology with explicit order semantics.
    pub fn with_order(n_objects: usize, order: InteractionOrder) -> Self {
        Self {
            n_objects,
            order,
            id_of_nodes: AHashMap::new(),
            nodes_of_id: Vec::new(),
            free_ids: Vec::new(),
        }
    }

    /// Returns the exclusive object-index bound.
    pub fn n_objects(&self) -> usize {
        self.n_objects
    }

    /// Returns current interaction ordering semantics.
    pub fn order(&self) -> InteractionOrder {
        self.order
    }

    /// Returns number of active interactions.
    pub fn len(&self) -> usize {
        self.id_of_nodes.len()
    }

    /// Returns true when no active interactions exist.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns total allocated id slots, including inactive reusable slots.
    pub fn capacity_slots(&self) -> usize {
        self.nodes_of_id.len()
    }

    /// Returns number of reusable inactive id slots.
    pub fn free_slot_count(&self) -> usize {
        self.free_ids.len()
    }

    /// Changes ordering semantics and rebuilds active lookup keys.
    ///
    /// Ordered-to-unordered conversion can fail if two different ordered
    /// interactions canonicalize to the same unordered node list.
    pub fn set_order(&mut self, order: InteractionOrder) -> Result<(), InteractionError> {
        if self.order == order {
            return Ok(());
        }

        let mut rebuilt = AHashMap::with_capacity(self.id_of_nodes.len());
        let mut rebuilt_nodes = self.nodes_of_id.clone();

        for (id, maybe_nodes) in rebuilt_nodes.iter_mut().enumerate() {
            let Some(nodes) = maybe_nodes else {
                continue;
            };

            let canonical = canonicalize_nodes(&nodes.nodes, order);
            let canonical_nodes = InteractionNodes::from_slice(&canonical);
            if let Some(&existing) = rebuilt.get(&canonical_nodes) {
                return Err(InteractionError::OrderChangeCollision {
                    nodes: canonical_nodes.nodes,
                    existing,
                    incoming: id,
                });
            }

            *nodes = canonical_nodes.clone();
            rebuilt.insert(canonical_nodes, id);
        }

        self.order = order;
        self.id_of_nodes = rebuilt;
        self.nodes_of_id = rebuilt_nodes;
        Ok(())
    }

    /// Strictly changes the object-index bound.
    ///
    /// Shrinking fails if an active interaction references an object that would
    /// fall outside the new range.
    pub fn set_n_objects(&mut self, n_objects: usize) -> Result<(), InteractionError> {
        if n_objects >= self.n_objects {
            self.n_objects = n_objects;
            return Ok(());
        }

        for (id, nodes) in self.iter() {
            if let Some(&obj) = nodes.nodes.iter().find(|&&obj| obj >= n_objects) {
                return Err(InteractionError::ObjectCountWouldInvalidate { n_objects, id, obj });
            }
        }

        self.n_objects = n_objects;
        Ok(())
    }

    /// Reserves storage for at least `additional` more interactions.
    pub fn reserve(&mut self, additional: usize) {
        self.id_of_nodes.reserve(additional);
        self.nodes_of_id.reserve(additional);
    }

    /// Shrinks or expands the object-index bound and removes now-invalid entries.
    ///
    /// Returns the ids removed because they referenced object indices outside the
    /// new bound.
    pub fn prune_n_objects(&mut self, n_objects: usize) -> Vec<InteractionId> {
        self.n_objects = n_objects;

        let ids_to_remove = self
            .nodes_of_id
            .iter()
            .enumerate()
            .filter_map(|(id, maybe_nodes)| {
                let nodes = maybe_nodes.as_ref()?;
                nodes
                    .nodes
                    .iter()
                    .any(|&obj| obj >= n_objects)
                    .then_some(id)
            })
            .collect::<Vec<_>>();

        for id in ids_to_remove.iter().copied() {
            self.remove_by_id_unchecked(id);
        }

        ids_to_remove
    }

    /// Returns nodes for an active interaction id.
    pub fn nodes_of(&self, id: InteractionId) -> Result<&InteractionNodes, InteractionError> {
        self.nodes_of_id
            .get(id)
            .and_then(|nodes| nodes.as_ref())
            .ok_or(InteractionError::InvalidInteractionId {
                id,
                n_slots: self.nodes_of_id.len(),
            })
    }

    /// Looks up the active interaction id for a node list.
    pub fn id_of(&self, nodes: &[ObjId]) -> Result<Option<InteractionId>, InteractionError> {
        let nodes = self.nodes_from_slice(nodes)?;
        Ok(self.id_of_nodes.get(&nodes).copied())
    }

    /// Returns whether the node list is active.
    pub fn contains(&self, nodes: &[ObjId]) -> Result<bool, InteractionError> {
        Ok(self.id_of(nodes)?.is_some())
    }

    /// Adds a node list if absent and returns its interaction id.
    pub fn add(&mut self, nodes: &[ObjId]) -> Result<InteractionId, InteractionError> {
        let nodes = self.nodes_from_slice(nodes)?;
        Ok(self.add_nodes(nodes))
    }

    fn add_nodes(&mut self, nodes: InteractionNodes) -> InteractionId {
        if let Some(&id) = self.id_of_nodes.get(&nodes) {
            return id;
        }

        let id = if let Some(id) = self.free_ids.pop() {
            self.nodes_of_id[id] = Some(nodes.clone());
            id
        } else {
            let id = self.nodes_of_id.len();
            self.nodes_of_id.push(Some(nodes.clone()));
            id
        };

        self.id_of_nodes.insert(nodes, id);
        id
    }

    /// Removes a node list if active and returns the released interaction id.
    pub fn remove(&mut self, nodes: &[ObjId]) -> Result<Option<InteractionId>, InteractionError> {
        let nodes = self.nodes_from_slice(nodes)?;
        Ok(self.remove_nodes(nodes))
    }

    fn remove_nodes(&mut self, nodes: InteractionNodes) -> Option<InteractionId> {
        let id = self.id_of_nodes.remove(&nodes)?;

        self.nodes_of_id[id] = None;
        self.free_ids.push(id);
        Some(id)
    }

    /// Clears all active interactions while preserving allocated id capacity.
    pub fn clear(&mut self) {
        self.id_of_nodes.clear();
        self.free_ids.clear();
        self.free_ids.extend((0..self.nodes_of_id.len()).rev());
        for slot in self.nodes_of_id.iter_mut() {
            *slot = None;
        }
    }

    /// Iterates active `(id, nodes)` entries.
    pub fn iter(&self) -> impl Iterator<Item = (InteractionId, &InteractionNodes)> + '_ {
        self.nodes_of_id
            .iter()
            .enumerate()
            .filter_map(|(id, nodes)| nodes.as_ref().map(|nodes| (id, nodes)))
    }

    /// Convenience pair lookup helper.
    pub fn id_of_pair(
        &self,
        i: ObjId,
        j: ObjId,
    ) -> Result<Option<InteractionId>, InteractionError> {
        self.validate_obj(i)?;
        self.validate_obj(j)?;
        let nodes = if self.order == InteractionOrder::Unordered && j < i {
            [j, i]
        } else {
            [i, j]
        };
        Ok(self.id_of_nodes.get(nodes.as_slice()).copied())
    }

    /// Convenience pair add helper.
    pub fn add_pair(&mut self, i: ObjId, j: ObjId) -> Result<InteractionId, InteractionError> {
        if let Some(id) = self.id_of_pair(i, j)? {
            return Ok(id);
        }
        let nodes = self.nodes_from_pair(i, j)?;
        Ok(self.add_nodes(nodes))
    }

    /// Convenience pair remove helper.
    pub fn remove_pair(
        &mut self,
        i: ObjId,
        j: ObjId,
    ) -> Result<Option<InteractionId>, InteractionError> {
        let nodes = self.nodes_from_pair(i, j)?;
        Ok(self.remove_nodes(nodes))
    }

    fn nodes_from_slice(&self, nodes: &[ObjId]) -> Result<InteractionNodes, InteractionError> {
        self.validate_nodes(nodes)?;
        Ok(InteractionNodes::from_slice(&canonicalize_nodes(
            nodes, self.order,
        )))
    }

    fn nodes_from_pair(&self, i: ObjId, j: ObjId) -> Result<InteractionNodes, InteractionError> {
        self.validate_obj(i)?;
        self.validate_obj(j)?;

        let (i, j) = if self.order == InteractionOrder::Unordered && j < i {
            (j, i)
        } else {
            (i, j)
        };
        Ok(InteractionNodes::from_pair(i, j))
    }

    fn validate_nodes(&self, nodes: &[ObjId]) -> Result<(), InteractionError> {
        if nodes.is_empty() {
            return Err(InteractionError::EmptyNodes);
        }

        for &obj in nodes {
            self.validate_obj(obj)?;
        }

        Ok(())
    }

    fn validate_obj(&self, obj: ObjId) -> Result<(), InteractionError> {
        if obj >= self.n_objects {
            return Err(InteractionError::InvalidObjId {
                obj,
                n_objects: self.n_objects,
            });
        }

        Ok(())
    }

    fn remove_by_id_unchecked(&mut self, id: InteractionId) {
        if let Some(nodes) = self.nodes_of_id.get_mut(id).and_then(|slot| slot.take()) {
            self.id_of_nodes.remove(&nodes);
            self.free_ids.push(id);
        }
    }
}

fn canonicalize_nodes(nodes: &[ObjId], order: InteractionOrder) -> Vec<ObjId> {
    if order == InteractionOrder::Unordered {
        let mut canonical = nodes.to_vec();
        canonical.sort_unstable();
        canonical
    } else {
        nodes.to_vec()
    }
}

/// Slot-indexed payload container with hole reuse.
#[derive(Debug, Clone)]
struct PayloadStore<T> {
    /// Slot payloads; `None` means this slot is currently free.
    slots: Vec<Option<T>>,
    /// Reusable free ids for O(1)-average insert/remove churn.
    free_ids: Vec<InteractionId>,
    /// Number of active payload entries currently stored.
    active_count: usize,
}

impl<T> Default for PayloadStore<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free_ids: Vec::new(),
            active_count: 0,
        }
    }
}

impl<T> PayloadStore<T> {
    /// Constructs an empty payload store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns number of allocated payload slots.
    pub fn capacity_slots(&self) -> usize {
        self.slots.len()
    }

    /// Reserves storage for at least `additional` more payload slots.
    pub fn reserve(&mut self, additional: usize) {
        self.slots.reserve(additional);
    }

    /// Inserts or overwrites payload at interaction id.
    pub fn set(&mut self, id: InteractionId, payload: T) {
        if id >= self.slots.len() {
            let old_len = self.slots.len();
            self.slots.resize_with(id + 1, || None);
            self.free_ids.extend((old_len..id).rev());
        }

        if self.slots[id].is_none() {
            self.active_count += 1;
            self.retain_free_id(id);
        }
        self.slots[id] = Some(payload);
    }

    /// Removes payload at interaction id and returns removed payload when present.
    pub fn remove(&mut self, id: InteractionId) -> Option<T> {
        let slot = self.slots.get_mut(id)?;
        let removed = slot.take();
        if removed.is_some() {
            self.active_count -= 1;
            self.free_ids.push(id);
        }
        removed
    }

    /// Returns immutable payload reference for active interaction id.
    pub fn get(&self, id: InteractionId) -> Option<&T> {
        self.slots.get(id).and_then(|x| x.as_ref())
    }

    /// Returns mutable payload reference for active interaction id.
    pub fn get_mut(&mut self, id: InteractionId) -> Option<&mut T> {
        self.slots.get_mut(id).and_then(|x| x.as_mut())
    }

    /// Clears all payload slots and marks everything reusable.
    pub fn clear(&mut self) {
        self.active_count = 0;
        self.free_ids.clear();
        self.free_ids.extend((0..self.slots.len()).rev());
        for slot in self.slots.iter_mut() {
            *slot = None;
        }
    }

    /// Parallel read-only visit over active payloads.
    pub fn par_for_each<F>(&self, f: F)
    where
        T: Sync,
        F: Fn(InteractionId, &T) + Send + Sync,
    {
        self.slots
            .par_iter()
            .with_min_len(crate::threading::parallel_chunk_len(self.slots.len()).unwrap_or(1))
            .enumerate()
            .for_each(|(id, slot)| {
                if let Some(payload) = slot.as_ref() {
                    f(id, payload);
                }
            });
    }

    /// Parallel mutable visit over active payloads.
    pub fn par_for_each_mut<F>(&mut self, f: F)
    where
        T: Send,
        F: Fn(InteractionId, &mut T) + Send + Sync,
    {
        let chunk = crate::threading::parallel_chunk_len(self.slots.len()).unwrap_or(1);
        self.slots
            .par_iter_mut()
            .with_min_len(chunk)
            .enumerate()
            .for_each(|(id, slot)| {
                if let Some(payload) = slot.as_mut() {
                    f(id, payload);
                }
            });
    }

    fn retain_free_id(&mut self, id: InteractionId) {
        if let Some(pos) = self.free_ids.iter().position(|&x| x == id) {
            self.free_ids.swap_remove(pos);
        }
    }
}

/// Synchronized topology + payload backend for one uniform payload type.
#[derive(Debug, Clone)]
pub struct Interaction<T> {
    /// Participant topology and id lifecycle state.
    topology: InteractionTopology,
    /// Payload storage synchronized with topology ids.
    payloads: PayloadStore<T>,
}

impl<T> Interaction<T> {
    /// Constructs an empty interaction table from object bound and order semantics.
    pub fn new(n_objects: usize, order: InteractionOrder) -> Self {
        Self {
            topology: InteractionTopology::with_order(n_objects, order),
            payloads: PayloadStore::new(),
        }
    }

    /// Constructs an interaction table from an existing topology.
    ///
    /// This constructor is intended for building payloads after a topology has
    /// been prepared. Existing topology ids will not have payloads until `set`
    /// is called for the matching node lists.
    pub fn with_topology(topology: InteractionTopology) -> Self {
        Self {
            topology,
            payloads: PayloadStore::new(),
        }
    }

    /// Returns immutable topology view.
    pub fn topology(&self) -> &InteractionTopology {
        &self.topology
    }

    /// Returns mutable topology view for advanced topology-only operations.
    ///
    /// Prefer `set_order`, `set_n_objects`, and `prune_n_objects` when payloads
    /// are present, because those methods keep payload storage synchronized.
    pub fn topology_mut(&mut self) -> &mut InteractionTopology {
        &mut self.topology
    }

    /// Returns number of active payload-backed interactions.
    pub fn len(&self) -> usize {
        self.payloads.active_count
    }

    /// Returns true when no payload-backed interactions exist.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the exclusive object-index bound.
    pub fn n_objects(&self) -> usize {
        self.topology.n_objects()
    }

    /// Returns current interaction ordering semantics.
    pub fn order(&self) -> InteractionOrder {
        self.topology.order()
    }

    /// Returns allocated id capacity.
    pub fn capacity_slots(&self) -> usize {
        self.topology.capacity_slots()
    }

    /// Reserves storage for at least `additional` more interactions and payloads.
    pub fn reserve(&mut self, additional: usize) {
        self.topology.reserve(additional);
        self.payloads.reserve(additional);
    }

    /// Changes ordering semantics while preserving synchronized payload ids.
    pub fn set_order(&mut self, order: InteractionOrder) -> Result<(), InteractionError> {
        self.topology.set_order(order)
    }

    /// Strictly changes object count and fails if any active interaction becomes invalid.
    pub fn set_n_objects(&mut self, n_objects: usize) -> Result<(), InteractionError> {
        self.topology.set_n_objects(n_objects)
    }

    /// Changes object count and removes interactions that reference invalid objects.
    ///
    /// Returns removed `(id, payload)` pairs.
    pub fn prune_n_objects(&mut self, n_objects: usize) -> Vec<(InteractionId, T)> {
        self.topology
            .prune_n_objects(n_objects)
            .into_iter()
            .filter_map(|id| self.payloads.remove(id).map(|payload| (id, payload)))
            .collect()
    }

    /// Returns whether a node list has an active payload-backed interaction.
    pub fn contains(&self, nodes: &[ObjId]) -> Result<bool, InteractionError> {
        let Some(id) = self.topology.id_of(nodes)? else {
            return Ok(false);
        };
        Ok(self.payloads.get(id).is_some())
    }

    /// Adds or overwrites one payload by node list and returns its interaction id.
    pub fn set(&mut self, nodes: &[ObjId], payload: T) -> Result<InteractionId, InteractionError> {
        let id = self.topology.add(nodes)?;
        self.payloads.set(id, payload);
        Ok(id)
    }

    /// Removes a node list and payload, returning `(id, payload)` when found.
    pub fn remove(
        &mut self,
        nodes: &[ObjId],
    ) -> Result<Option<(InteractionId, T)>, InteractionError> {
        let Some(id) = self.topology.remove(nodes)? else {
            return Ok(None);
        };
        let payload = self
            .payloads
            .remove(id)
            .ok_or(InteractionError::InvalidInteractionId {
                id,
                n_slots: self.payloads.capacity_slots(),
            })?;
        Ok(Some((id, payload)))
    }

    /// Returns immutable payload reference by node list.
    pub fn get(&self, nodes: &[ObjId]) -> Result<Option<&T>, InteractionError> {
        let Some(id) = self.topology.id_of(nodes)? else {
            return Ok(None);
        };
        Ok(self.payloads.get(id))
    }

    /// Returns mutable payload reference by node list.
    pub fn get_mut(&mut self, nodes: &[ObjId]) -> Result<Option<&mut T>, InteractionError> {
        let Some(id) = self.topology.id_of(nodes)? else {
            return Ok(None);
        };
        Ok(self.payloads.get_mut(id))
    }

    /// Adds or overwrites one pair payload.
    pub fn set_pair(
        &mut self,
        i: ObjId,
        j: ObjId,
        payload: T,
    ) -> Result<InteractionId, InteractionError> {
        let id = self.topology.add_pair(i, j)?;
        self.payloads.set(id, payload);
        Ok(id)
    }

    /// Returns immutable payload reference by pair.
    pub fn get_pair(&self, i: ObjId, j: ObjId) -> Result<Option<&T>, InteractionError> {
        Ok(self
            .topology
            .id_of_pair(i, j)?
            .and_then(|id| self.payloads.get(id)))
    }

    /// Returns mutable payload reference by pair.
    pub fn get_pair_mut(&mut self, i: ObjId, j: ObjId) -> Result<Option<&mut T>, InteractionError> {
        Ok(self
            .topology
            .id_of_pair(i, j)?
            .and_then(|id| self.payloads.get_mut(id)))
    }

    /// Removes one pair payload.
    pub fn remove_pair(
        &mut self,
        i: ObjId,
        j: ObjId,
    ) -> Result<Option<(InteractionId, T)>, InteractionError> {
        self.remove(&[i, j])
    }

    /// Clears both topology and payload storage.
    pub fn clear(&mut self) {
        self.topology.clear();
        self.payloads.clear();
    }

    /// Iterates active `(id, nodes, payload)` entries.
    pub fn iter(&self) -> impl Iterator<Item = (InteractionId, &InteractionNodes, &T)> {
        self.topology
            .iter()
            .filter_map(|(id, nodes)| self.payloads.get(id).map(|payload| (id, nodes, payload)))
    }

    /// Parallel mutable payload visit over active interactions.
    pub fn par_for_each_payload_mut<F>(&mut self, f: F)
    where
        T: Send,
        F: Fn(InteractionId, &mut T) + Send + Sync,
    {
        self.payloads.par_for_each_mut(f);
    }

    /// Parallel read-only payload visit over active interactions.
    pub fn par_for_each_payload<F>(&self, f: F)
    where
        T: Sync,
        F: Fn(InteractionId, &T) + Send + Sync,
    {
        self.payloads.par_for_each(f);
    }

    /// Parallel read-only visit over active `(id, nodes, payload)` entries.
    pub fn par_for_each<F>(&self, f: F)
    where
        T: Sync,
        F: Fn(InteractionId, &InteractionNodes, &T) + Send + Sync,
    {
        self.payloads.par_for_each(|id, payload| {
            let nodes = self.topology.nodes_of(id).expect(
                "interaction topology/payload storage out of sync while parallel iterating active entries",
            );
            f(id, nodes, payload);
        });
    }
}

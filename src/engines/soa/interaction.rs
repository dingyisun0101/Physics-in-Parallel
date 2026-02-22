/*!
Core n-point interaction backend for `engines::soa`.

This module provides:
- `Topology`: key-to-slot mapping for mixed-arity interactions.
- `PayloadStore<T>`: slot-indexed payload storage with hole reuse.
- `Interaction<T>`: synchronized topology + payload container.
*/

use ahash::AHashMap;
use rayon::prelude::*;

use super::phys_obj::AttrsError;

/// Stable object index in `PhysObj` columns.
pub type ObjId = usize;
/// Stable interaction slot index.
pub type EdgeId = usize;

/// Key ordering mode used by topology validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectionMode {
    /// Preserve caller-provided node order exactly.
    Directed,
    /// Require nondecreasing node order (`nodes[k-1] <= nodes[k]`).
    Undirected,
}

/// Hashable mixed-arity interaction key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InteractionKey {
    /// Object indices identifying one interaction term; arity is `nodes.len()`.
    pub nodes: Box<[ObjId]>,
}

impl InteractionKey {
    /// Builds an owned interaction key from node indices.
    pub fn from_slice(nodes: &[ObjId]) -> Self {
        Self {
            nodes: nodes.into(),
        }
    }

    /// Returns interaction arity (`nodes.len()`).
    pub fn arity(&self) -> usize {
        self.nodes.len()
    }
}

/// Errors returned by interaction backend operations.
#[derive(Debug, Clone, PartialEq)]
pub enum InteractionError {
    /// Object index is outside topology bounds.
    InvalidObjId {
        /// Invalid object index value.
        obj: ObjId,
        /// Current exclusive upper bound (`0..n_objects`).
        n_objects: usize,
    },
    /// Undirected mode received a non-canonical node ordering.
    InvalidUndirectedOrder {
        /// Position where ordering check failed.
        at: usize,
        /// Previous node at `at - 1`.
        prev: ObjId,
        /// Current node at `at`.
        curr: ObjId,
    },
    /// Slot id does not exist or is inactive.
    InvalidEdgeId {
        /// Requested slot id.
        edge: EdgeId,
        /// Total slot capacity at time of check.
        n_slots: usize,
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

/// Mixed-arity key index mapping interaction keys to stable slot ids.
#[derive(Debug, Clone)]
pub struct Topology {
    /// Exclusive upper bound for valid object indices (`0..n_objects`).
    n_objects: usize,
    /// Key-ordering mode used by validation.
    mode: DirectionMode,
    /// Forward mapping from interaction key to stable slot id.
    slot_of_key: AHashMap<InteractionKey, EdgeId>,
    /// Reverse mapping from slot id to key; `None` means slot is currently free.
    key_of_slot: Vec<Option<InteractionKey>>,
    /// Reusable free slot ids used for O(1)-average insert/delete churn.
    free_slots: Vec<EdgeId>,
}

impl Topology {
    /// Constructs an empty topology in undirected mode.
    pub fn new(n_objects: usize) -> Self {
        Self::with_mode(n_objects, DirectionMode::Undirected)
    }

    /// Constructs an empty topology with explicit direction mode.
    pub fn with_mode(n_objects: usize, mode: DirectionMode) -> Self {
        Self {
            n_objects,
            mode,
            slot_of_key: AHashMap::new(),
            key_of_slot: Vec::new(),
            free_slots: Vec::new(),
        }
    }

    /// Returns object-index bound.
    pub fn n_objects(&self) -> usize {
        self.n_objects
    }

    /// Returns current direction mode.
    pub fn mode(&self) -> DirectionMode {
        self.mode
    }

    /// Sets direction mode for future operations.
    pub fn set_mode(&mut self, mode: DirectionMode) {
        self.mode = mode;
    }

    /// Sets object-index bound for future validation.
    pub fn set_n_objects(&mut self, n_objects: usize) {
        self.n_objects = n_objects;
    }

    /// Returns total slot capacity (active + free).
    pub fn len_slots(&self) -> usize {
        self.key_of_slot.len()
    }

    /// Returns number of active mapped interaction keys.
    pub fn active_count(&self) -> usize {
        self.slot_of_key.len()
    }

    /// Returns number of currently free slots.
    pub fn free_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Returns key for active slot id.
    pub fn edge_key(&self, edge: EdgeId) -> Result<&InteractionKey, InteractionError> {
        self.key_of_slot
            .get(edge)
            .and_then(|k| k.as_ref())
            .ok_or(InteractionError::InvalidEdgeId {
                edge,
                n_slots: self.key_of_slot.len(),
            })
    }

    /// Looks up active slot id for interaction key.
    pub fn index_of(&self, nodes: &[ObjId]) -> Result<Option<EdgeId>, InteractionError> {
        let key = self.key_from_nodes(nodes)?;
        Ok(self.slot_of_key.get(&key).copied())
    }

    /// Returns whether key exists in active mapping.
    pub fn contains_key(&self, nodes: &[ObjId]) -> Result<bool, InteractionError> {
        Ok(self.index_of(nodes)?.is_some())
    }

    /// Inserts key if absent and returns slot id.
    pub fn insert(&mut self, nodes: &[ObjId]) -> Result<EdgeId, InteractionError> {
        let key = self.key_from_nodes(nodes)?;
        if let Some(&edge) = self.slot_of_key.get(&key) {
            return Ok(edge);
        }

        let edge = if let Some(slot) = self.free_slots.pop() {
            self.key_of_slot[slot] = Some(key.clone());
            slot
        } else {
            let slot = self.key_of_slot.len();
            self.key_of_slot.push(Some(key.clone()));
            slot
        };

        self.slot_of_key.insert(key, edge);
        Ok(edge)
    }

    /// Removes key if active and returns released slot id.
    pub fn remove(&mut self, nodes: &[ObjId]) -> Result<Option<EdgeId>, InteractionError> {
        let key = self.key_from_nodes(nodes)?;
        let Some(edge) = self.slot_of_key.remove(&key) else {
            return Ok(None);
        };
        let n_slots = self.key_of_slot.len();

        let slot = self
            .key_of_slot
            .get_mut(edge)
            .ok_or(InteractionError::InvalidEdgeId { edge, n_slots })?;
        *slot = None;
        self.free_slots.push(edge);
        Ok(Some(edge))
    }

    /// Deletes key if active.
    pub fn delete(&mut self, nodes: &[ObjId]) -> Result<(), InteractionError> {
        let _ = self.remove(nodes)?;
        Ok(())
    }

    /// Clears all active mappings while preserving capacity.
    pub fn clear(&mut self) {
        self.slot_of_key.clear();
        self.free_slots.clear();
        self.free_slots.extend((0..self.key_of_slot.len()).rev());
        for slot in self.key_of_slot.iter_mut() {
            *slot = None;
        }
    }

    /// Iterates active `(slot, key)` entries.
    pub fn iter_active(&self) -> impl Iterator<Item = (EdgeId, &InteractionKey)> + '_ {
        self.key_of_slot
            .iter()
            .enumerate()
            .filter_map(|(edge, key)| key.as_ref().map(|k| (edge, k)))
    }

    /// Convenience pair lookup helper.
    pub fn index_of_pair(&self, i: ObjId, j: ObjId) -> Result<Option<EdgeId>, InteractionError> {
        self.index_of(&[i, j])
    }

    /// Convenience pair insert helper.
    pub fn insert_pair(&mut self, i: ObjId, j: ObjId) -> Result<EdgeId, InteractionError> {
        self.insert(&[i, j])
    }

    /// Convenience pair remove helper.
    pub fn remove_pair(&mut self, i: ObjId, j: ObjId) -> Result<Option<EdgeId>, InteractionError> {
        self.remove(&[i, j])
    }

    fn key_from_nodes(&self, nodes: &[ObjId]) -> Result<InteractionKey, InteractionError> {
        self.validate_nodes(nodes)?;
        Ok(InteractionKey::from_slice(nodes))
    }

    fn validate_nodes(&self, nodes: &[ObjId]) -> Result<(), InteractionError> {
        for &obj in nodes {
            if obj >= self.n_objects {
                return Err(InteractionError::InvalidObjId {
                    obj,
                    n_objects: self.n_objects,
                });
            }
        }

        if self.mode == DirectionMode::Undirected {
            for k in 1..nodes.len() {
                if nodes[k - 1] > nodes[k] {
                    return Err(InteractionError::InvalidUndirectedOrder {
                        at: k,
                        prev: nodes[k - 1],
                        curr: nodes[k],
                    });
                }
            }
        }

        Ok(())
    }
}

/// Slot-indexed payload container with hole reuse.
#[derive(Debug, Clone)]
struct PayloadStore<T> {
    /// Slot payloads; `None` means this slot is currently free (hole).
    slots: Vec<Option<T>>,
    /// Reusable free slot ids for O(1)-average insert/delete churn.
    free_slots: Vec<EdgeId>,
    /// Number of active payload entries (`Some`) currently stored.
    active_count: usize,
}

impl<T> Default for PayloadStore<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free_slots: Vec::new(),
            active_count: 0,
        }
    }
}

impl<T> PayloadStore<T> {
    /// Constructs an empty payload store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns number of allocated slots (active + free).
    pub fn len_slots(&self) -> usize {
        self.slots.len()
    }

    /// Inserts or overwrites payload at slot id.
    pub fn insert_or_assign(&mut self, edge: EdgeId, payload: T) {
        if edge >= self.slots.len() {
            let old_len = self.slots.len();
            self.slots.resize_with(edge + 1, || None);
            self.free_slots.extend((old_len..edge).rev());
        }

        if self.slots[edge].is_none() {
            self.active_count += 1;
            self.retain_free_slot(edge);
        }
        self.slots[edge] = Some(payload);
    }

    /// Removes payload at slot id and returns removed payload when present.
    pub fn remove(&mut self, edge: EdgeId) -> Option<T> {
        let slot = self.slots.get_mut(edge)?;
        let removed = slot.take();
        if removed.is_some() {
            self.active_count -= 1;
            self.free_slots.push(edge);
        }
        removed
    }

    /// Returns immutable payload reference for active slot id.
    pub fn get(&self, edge: EdgeId) -> Option<&T> {
        self.slots.get(edge).and_then(|x| x.as_ref())
    }

    /// Returns mutable payload reference for active slot id.
    pub fn get_mut(&mut self, edge: EdgeId) -> Option<&mut T> {
        self.slots.get_mut(edge).and_then(|x| x.as_mut())
    }

    /// Clears all payload slots and marks everything reusable.
    pub fn clear(&mut self) {
        self.active_count = 0;
        self.free_slots.clear();
        self.free_slots.extend((0..self.slots.len()).rev());
        for slot in self.slots.iter_mut() {
            *slot = None;
        }
    }

    /// Parallel read-only visit over active slot payloads.
    pub fn par_for_each_active<F>(&self, f: F)
    where
        T: Sync,
        F: Fn(EdgeId, &T) + Send + Sync,
    {
        self.slots.par_iter().enumerate().for_each(|(edge, slot)| {
            if let Some(payload) = slot.as_ref() {
                f(edge, payload);
            }
        });
    }

    /// Parallel mutable visit over active slot payloads.
    pub fn par_for_each_active_mut<F>(&mut self, f: F)
    where
        T: Send,
        F: Fn(EdgeId, &mut T) + Send + Sync,
    {
        self.slots
            .par_iter_mut()
            .enumerate()
            .for_each(|(edge, slot)| {
                if let Some(payload) = slot.as_mut() {
                    f(edge, payload);
                }
            });
    }

    fn retain_free_slot(&mut self, edge: EdgeId) {
        if let Some(pos) = self.free_slots.iter().position(|&x| x == edge) {
            self.free_slots.swap_remove(pos);
        }
    }
}

/// Synchronized topology + payload backend for one uniform payload type.
#[derive(Debug, Clone)]
pub struct Interaction<T> {
    /// Key-to-slot index and slot lifecycle state.
    topology: Topology,
    /// Slot-indexed payload storage synchronized with topology slots.
    payloads: PayloadStore<T>,
}

impl<T> Interaction<T> {
    /// Constructs table from an existing topology.
    pub fn with_topology(topology: Topology) -> Self {
        Self {
            topology,
            payloads: PayloadStore::new(),
        }
    }

    /// Constructs table from object bound and direction mode.
    pub fn new(n_objects: usize, mode: DirectionMode) -> Self {
        Self::with_topology(Topology::with_mode(n_objects, mode))
    }

    /// Returns immutable topology view.
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// Returns mutable topology view.
    pub fn topology_mut(&mut self) -> &mut Topology {
        &mut self.topology
    }

    /// Returns whether key exists.
    pub fn contains_key(&self, nodes: &[ObjId]) -> Result<bool, InteractionError> {
        self.topology.contains_key(nodes)
    }

    /// Inserts key and payload, returning slot id.
    pub fn insert(&mut self, nodes: &[ObjId], payload: T) -> Result<EdgeId, InteractionError> {
        let edge = self.topology.insert(nodes)?;
        self.payloads.insert_or_assign(edge, payload);
        Ok(edge)
    }

    /// Removes key and payload, returning `(slot, payload)` when found.
    pub fn remove(&mut self, nodes: &[ObjId]) -> Result<Option<(EdgeId, T)>, InteractionError> {
        let Some(edge) = self.topology.remove(nodes)? else {
            return Ok(None);
        };
        let payload = self
            .payloads
            .remove(edge)
            .ok_or(InteractionError::InvalidEdgeId {
                edge,
                n_slots: self.payloads.len_slots(),
            })?;
        Ok(Some((edge, payload)))
    }

    /// Returns immutable payload reference by key.
    pub fn get(&self, nodes: &[ObjId]) -> Result<Option<&T>, InteractionError> {
        let Some(edge) = self.topology.index_of(nodes)? else {
            return Ok(None);
        };
        Ok(self.payloads.get(edge))
    }

    /// Returns mutable payload reference by key.
    pub fn get_mut(&mut self, nodes: &[ObjId]) -> Result<Option<&mut T>, InteractionError> {
        let Some(edge) = self.topology.index_of(nodes)? else {
            return Ok(None);
        };
        Ok(self.payloads.get_mut(edge))
    }

    /// Clears both topology and payload storage.
    pub fn clear(&mut self) {
        self.topology.clear();
        self.payloads.clear();
    }

    /// Iterates active `(slot, key, payload)` entries.
    pub fn iter_active(&self) -> impl Iterator<Item = (EdgeId, &InteractionKey, &T)> {
        self.topology
            .iter_active()
            .filter_map(|(edge, key)| self.payloads.get(edge).map(|payload| (edge, key, payload)))
    }

    /// Parallel mutable payload visit over active slots.
    pub fn par_for_each_active_payload_mut<F>(&mut self, f: F)
    where
        T: Send,
        F: Fn(EdgeId, &mut T) + Send + Sync,
    {
        self.payloads.par_for_each_active_mut(f);
    }

    /// Parallel read-only payload visit over active slots.
    pub fn par_for_each_active_payload<F>(&self, f: F)
    where
        T: Sync,
        F: Fn(EdgeId, &T) + Send + Sync,
    {
        self.payloads.par_for_each_active(f);
    }
}

/*!
`interaction` defines general pair-interaction primitives for `engines::soa`.

Scope of this module
--------------------
1. `PairTopology`: undirected pair-graph over object IDs with fast membership checks.
2. `Interaction` trait: generic stepping contract for force/acceleration accumulation.

Design notes
------------
- Pair storage is canonicalized as `(min(i,j), max(i,j))`.
- A hash table stores live-edge membership for O(1) lookup.
- An explicit `active_edges` list avoids hash iteration in physics loops.
- `to_adj_matrix()` exports a standard adjacency matrix for debugging/analysis.
*/

use ahash::AHashMap;

use crate::math::tensor::rank_2::matrix::dense::Matrix;

use super::phys_obj::{ObjId, PhysObj, PhysObjError};

// ======================================================================================
// ---------------------------------- Type aliases --------------------------------------
// ======================================================================================

/// Stable edge index in topology-owned arrays.
pub type EdgeId = usize;

// ======================================================================================
// ------------------------------------- Errors -----------------------------------------
// ======================================================================================

/// Errors for interaction/topology operations.
#[derive(Debug, Clone, PartialEq)]
pub enum InteractionError {
    /// Object ID is outside topology bounds.
    InvalidObjId { obj: ObjId, n_objects: usize },
    /// A pair `(i, i)` is not a valid edge in this undirected topology.
    SelfEdge { obj: ObjId },
    /// Edge index does not exist.
    InvalidEdgeId { edge: EdgeId, n_edges: usize },
    /// Lifted error from `PhysObj` accessors.
    PhysObj(PhysObjError),
}

impl From<PhysObjError> for InteractionError {
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `from` logic for this module.
    /// - Parameters:
    ///   - `value` (`PhysObjError`): Value provided by caller for write/update behavior.
    fn from(value: PhysObjError) -> Self {
        Self::PhysObj(value)
    }
}

// ======================================================================================
// --------------------------------- Pair Topology --------------------------------------
// ======================================================================================

/// Pack canonicalized `(i, j)` into one `u64` key for hash membership.
#[inline]
/// Annotation:
/// - Purpose: Executes `pack` logic for this module.
/// - Parameters:
///   - `i` (`ObjId`): Primary index argument.
///   - `j` (`ObjId`): Secondary index argument.
fn pack(i: ObjId, j: ObjId) -> u64 {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    ((a as u64) << 32) | (b as u64)
}

/// Canonicalize `(i, j)` as `(min(i,j), max(i,j))`.
#[inline]
/// Annotation:
/// - Purpose: Executes `canonical_pair` logic for this module.
/// - Parameters:
///   - `i` (`ObjId`): Primary index argument.
///   - `j` (`ObjId`): Secondary index argument.
fn canonical_pair(i: ObjId, j: ObjId) -> (ObjId, ObjId) {
    if i < j { (i, j) } else { (j, i) }
}

/**
Undirected pair topology for object interactions.

Storage
-------
- `edges[e] = (i, j)` stores canonical endpoint pair for edge `e`.
- `table[pack(i,j)] = e` stores membership of active edges.
- `active_edges` stores live edge IDs for hot-loop iteration.
- `pos_in_active[e]` stores `active_edges` position or `None` if inactive.
*/
#[derive(Debug, Clone)]
pub struct PairTopology {
    n_objects: usize,
    edges: Vec<(ObjId, ObjId)>,
    table: AHashMap<u64, EdgeId>,
    active_edges: Vec<EdgeId>,
    pos_in_active: Vec<Option<usize>>,
}

impl PairTopology {
    /// Create an empty topology over object IDs `[0, n_objects)`.
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `n_objects` (`usize`): Object-state container used by this operation.
    pub fn new(n_objects: usize) -> Self {
        assert!(n_objects > 0, "PairTopology::new: n_objects must be > 0");
        Self {
            n_objects,
            edges: Vec::new(),
            table: AHashMap::new(),
            active_edges: Vec::new(),
            pos_in_active: Vec::new(),
        }
    }

    /// Number of objects addressable by this topology.
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `n_objects` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn n_objects(&self) -> usize {
        self.n_objects
    }

    /// Update object-bound metadata for future inserts/lookups.
    ///
    /// Existing edge IDs remain unchanged. Caller should guarantee any existing edges are
    /// still valid under the new bound.
    #[inline]
    /// Annotation:
    /// - Purpose: Sets the `n_objects` value.
    /// - Parameters:
    ///   - `n_objects` (`usize`): Object-state container used by this operation.
    pub fn set_n_objects(&mut self, n_objects: usize) {
        assert!(n_objects > 0, "PairTopology::set_n_objects: n_objects must be > 0");
        self.n_objects = n_objects;
    }

    /// Total number of edge slots ever created (active + inactive).
    #[inline]
    /// Annotation:
    /// - Purpose: Returns the current length/size.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn len_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of active edges.
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `active_count` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn active_count(&self) -> usize {
        self.active_edges.len()
    }

    /// Borrow active edge list.
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `active_edges` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn active_edges(&self) -> &[EdgeId] {
        &self.active_edges
    }

    /// Get endpoints for edge `e`.
    /// Annotation:
    /// - Purpose: Executes `edge_pair` logic for this module.
    /// - Parameters:
    ///   - `e` (`EdgeId`): Parameter of type `EdgeId` used by `edge_pair`.
    pub fn edge_pair(&self, e: EdgeId) -> Result<(ObjId, ObjId), InteractionError> {
        self.edges
            .get(e)
            .copied()
            .ok_or(InteractionError::InvalidEdgeId {
                edge: e,
                n_edges: self.edges.len(),
            })
    }

    /// Get active edge ID for pair `(i,j)`, if present.
    #[inline]
    /// Annotation:
    /// - Purpose: Computes an index mapping for input coordinates.
    /// - Parameters:
    ///   - `i` (`ObjId`): Primary index argument.
    ///   - `j` (`ObjId`): Secondary index argument.
    pub fn index_of(&self, i: ObjId, j: ObjId) -> Option<EdgeId> {
        self.table.get(&pack(i, j)).copied()
    }

    /// Fast membership check for active pair.
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `contains_pair` logic for this module.
    /// - Parameters:
    ///   - `i` (`ObjId`): Primary index argument.
    ///   - `j` (`ObjId`): Secondary index argument.
    pub fn contains_pair(&self, i: ObjId, j: ObjId) -> bool {
        self.table.contains_key(&pack(i, j))
    }

    /// True iff edge index exists and is active.
    #[inline]
    /// Annotation:
    /// - Purpose: Checks whether `active_index` condition is true.
    /// - Parameters:
    ///   - `e` (`EdgeId`): Parameter of type `EdgeId` used by `is_active_index`.
    pub fn is_active_index(&self, e: EdgeId) -> bool {
        self.pos_in_active.get(e).and_then(|x| *x).is_some()
    }

    /// Insert pair `(i, j)` if absent, otherwise return existing edge ID.
    /// Annotation:
    /// - Purpose: Inserts data into the underlying structure.
    /// - Parameters:
    ///   - `i` (`ObjId`): Primary index argument.
    ///   - `j` (`ObjId`): Secondary index argument.
    pub fn insert(&mut self, i: ObjId, j: ObjId) -> Result<EdgeId, InteractionError> {
        self.validate_pair(i, j)?;

        let key = pack(i, j);
        if let Some(&e) = self.table.get(&key) {
            return Ok(e);
        }

        let (a, b) = canonical_pair(i, j);
        let e = self.edges.len();
        self.edges.push((a, b));
        self.table.insert(key, e);

        let pos = self.active_edges.len();
        self.active_edges.push(e);
        self.pos_in_active.push(Some(pos));

        Ok(e)
    }

    /// Delete pair `(i, j)` if active.
    /// Annotation:
    /// - Purpose: Removes data from the underlying structure.
    /// - Parameters:
    ///   - `i` (`ObjId`): Primary index argument.
    ///   - `j` (`ObjId`): Secondary index argument.
    pub fn delete(&mut self, i: ObjId, j: ObjId) -> Result<(), InteractionError> {
        self.validate_pair(i, j)?;

        let key = pack(i, j);
        let Some(&e) = self.table.get(&key) else {
            return Ok(());
        };
        self.table.remove(&key);

        if let Some(pos) = self.pos_in_active[e] {
            let last_pos = self.active_edges.len() - 1;
            self.active_edges.swap(pos, last_pos);
            let moved_e = self.active_edges[pos];
            self.pos_in_active[moved_e] = Some(pos);
            self.active_edges.pop();
            self.pos_in_active[e] = None;
        }

        Ok(())
    }

    /**
    Build adjacency matrix `A` of shape `[n_objects, n_objects]`.

    Semantics
    ---------
    - `A[i, j] = 1` iff edge `(i, j)` is active, else `0`.
    - Matrix is symmetric by construction.
    - Diagonal is always `0`.
    */
    /// Annotation:
    /// - Purpose: Converts this value into `adj_matrix` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_adj_matrix(&self) -> Matrix<usize> {
        let mut a = Matrix::<usize>::empty(self.n_objects, self.n_objects);
        for &e in &self.active_edges {
            let (i, j) = self.edges[e];
            a.set(i as isize, j as isize, 1usize);
            a.set(j as isize, i as isize, 1usize);
        }
        a
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `validate_pair` logic for this module.
    /// - Parameters:
    ///   - `i` (`ObjId`): Primary index argument.
    ///   - `j` (`ObjId`): Secondary index argument.
    fn validate_pair(&self, i: ObjId, j: ObjId) -> Result<(), InteractionError> {
        if i >= self.n_objects {
            return Err(InteractionError::InvalidObjId {
                obj: i,
                n_objects: self.n_objects,
            });
        }
        if j >= self.n_objects {
            return Err(InteractionError::InvalidObjId {
                obj: j,
                n_objects: self.n_objects,
            });
        }
        if i == j {
            return Err(InteractionError::SelfEdge { obj: i });
        }
        Ok(())
    }
}

// ======================================================================================
// -------------------------------- Interaction trait -----------------------------------
// ======================================================================================

/// Minimal behavior for a pair interaction module that operates on `PhysObj`.
pub trait Interaction<const D: usize> {
    /// Immutable topology view.
    /// Annotation:
    /// - Purpose: Executes `topology` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn topology(&self) -> &PairTopology;

    /// Mutable topology view.
    /// Annotation:
    /// - Purpose: Executes `topology_mut` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn topology_mut(&mut self) -> &mut PairTopology;

    /// Optional topology update hook (default: no-op).
    /// Annotation:
    /// - Purpose: Updates `topology` state.
    /// - Parameters:
    ///   - `_objects` (`&PhysObj<D>`): Object-state container used by this operation.
    fn update_topology(&mut self, _objects: &PhysObj<D>) -> Result<(), InteractionError> {
        Ok(())
    }

    /// Accumulate acceleration contributions into `objects.acc`.
    /// Annotation:
    /// - Purpose: Executes `accumulate_acc` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj<D>`): Object-state container used by this operation.
    fn accumulate_acc(&self, objects: &mut PhysObj<D>) -> Result<(), InteractionError>;

    /// One interaction step = topology update + acceleration accumulation.
    /// Annotation:
    /// - Purpose: Executes `step` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj<D>`): Object-state container used by this operation.
    fn step(&mut self, objects: &mut PhysObj<D>) -> Result<(), InteractionError> {
        self.update_topology(objects)?;
        self.accumulate_acc(objects)
    }
}

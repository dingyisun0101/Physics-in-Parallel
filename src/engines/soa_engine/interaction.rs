/*!
`interaction` defines pairwise interaction primitives for `engines::soa_engine`.

Scope of this module
--------------------
1. `PairTopology`: undirected pair-graph over object IDs with fast membership checks.
2. `Interaction` trait: minimal stepping contract for force accumulation.
3. `SpringInteraction`: first concrete interaction law (Hooke springs on active edges).

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
    /// Non-finite or invalid physics parameter.
    InvalidParam { field: &'static str, value: f64 },
    /// Internal parameter vectors and edge arrays diverged in length.
    ParamLengthMismatch { n_edges: usize, n_k: usize, n_l0: usize },
    /// Lifted error from `PhysObj` accessors.
    PhysObj(PhysObjError),
}

impl From<PhysObjError> for InteractionError {
    #[inline]
    fn from(value: PhysObjError) -> Self {
        Self::PhysObj(value)
    }
}

// ======================================================================================
// --------------------------------- Pair Topology --------------------------------------
// ======================================================================================

/// Pack canonicalized `(i, j)` into one `u64` key for hash membership.
#[inline]
fn pack(i: ObjId, j: ObjId) -> u64 {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    ((a as u64) << 32) | (b as u64)
}

/// Canonicalize `(i, j)` as `(min(i,j), max(i,j))`.
#[inline]
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
    pub fn n_objects(&self) -> usize {
        self.n_objects
    }

    /// Update object-bound metadata for future inserts/lookups.
    ///
    /// Existing edge IDs remain unchanged. Caller should guarantee any existing edges are
    /// still valid under the new bound.
    #[inline]
    pub fn set_n_objects(&mut self, n_objects: usize) {
        assert!(n_objects > 0, "PairTopology::set_n_objects: n_objects must be > 0");
        self.n_objects = n_objects;
    }

    /// Total number of edge slots ever created (active + inactive).
    #[inline]
    pub fn len_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of active edges.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_edges.len()
    }

    /// Borrow active edge list.
    #[inline]
    pub fn active_edges(&self) -> &[EdgeId] {
        &self.active_edges
    }

    /// Get endpoints for edge `e`.
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
    pub fn index_of(&self, i: ObjId, j: ObjId) -> Option<EdgeId> {
        self.table.get(&pack(i, j)).copied()
    }

    /// Fast membership check for active pair.
    #[inline]
    pub fn contains_pair(&self, i: ObjId, j: ObjId) -> bool {
        self.table.contains_key(&pack(i, j))
    }

    /// True iff edge index exists and is active.
    #[inline]
    pub fn is_active_index(&self, e: EdgeId) -> bool {
        self.pos_in_active.get(e).and_then(|x| *x).is_some()
    }

    /// Insert pair `(i, j)` if absent, otherwise return existing edge ID.
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

/// Minimal behavior for an interaction module that can operate on `PhysObj`.
pub trait Interaction<const D: usize> {
    /// Immutable topology view.
    fn topology(&self) -> &PairTopology;
    /// Mutable topology view.
    fn topology_mut(&mut self) -> &mut PairTopology;

    /// Optional topology update hook (default: no-op).
    fn update_topology(&mut self, _objects: &PhysObj<D>) -> Result<(), InteractionError> {
        Ok(())
    }

    /// Accumulate acceleration contributions into `objects.acc`.
    fn accumulate_acc(&self, objects: &mut PhysObj<D>) -> Result<(), InteractionError>;

    /// One interaction step = topology update + acceleration accumulation.
    fn step(&mut self, objects: &mut PhysObj<D>) -> Result<(), InteractionError> {
        self.update_topology(objects)?;
        self.accumulate_acc(objects)
    }
}

// ======================================================================================
// ------------------------------ Spring interaction ------------------------------------
// ======================================================================================

/**
Hookean spring interaction on active topology edges.

Per-edge parameters
-------------------
- `k[e]`: spring constant (`> 0`, finite)
- `l0[e]`: rest length (`>= 0`, finite)
*/
#[derive(Debug, Clone)]
pub struct SpringInteraction {
    topology: PairTopology,
    k: Vec<f64>,
    l0: Vec<f64>,
}

impl SpringInteraction {
    /// Create an empty spring interaction for `n_objects` objects.
    pub fn new(n_objects: usize) -> Self {
        Self {
            topology: PairTopology::new(n_objects),
            k: Vec::new(),
            l0: Vec::new(),
        }
    }

    /// Construct from pre-built topology with uniform default parameters.
    pub fn with_topology(
        topology: PairTopology,
        k_default: f64,
        l0_default: f64,
    ) -> Result<Self, InteractionError> {
        Self::validate_param("k_default", k_default, true)?;
        Self::validate_param("l0_default", l0_default, false)?;

        let m = topology.len_edges();
        Ok(Self {
            topology,
            k: vec![k_default; m],
            l0: vec![l0_default; m],
        })
    }

    /// Number of total edge slots.
    #[inline]
    pub fn len_edges(&self) -> usize {
        self.topology.len_edges()
    }

    /// Number of active edges.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.topology.active_count()
    }

    /// Insert/update one spring edge.
    ///
    /// If `(i,j)` already exists and is active, this updates parameters in place.
    /// Otherwise a new edge slot is created.
    pub fn add_edge(
        &mut self,
        i: ObjId,
        j: ObjId,
        k: f64,
        l0: f64,
    ) -> Result<EdgeId, InteractionError> {
        Self::validate_param("k", k, true)?;
        Self::validate_param("l0", l0, false)?;

        if let Some(e) = self.topology.index_of(i, j) {
            self.k[e] = k;
            self.l0[e] = l0;
            return Ok(e);
        }

        let e = self.topology.insert(i, j)?;
        debug_assert_eq!(e, self.k.len());
        self.k.push(k);
        self.l0.push(l0);
        Ok(e)
    }

    /// Delete one spring pair if active.
    #[inline]
    pub fn remove_edge(&mut self, i: ObjId, j: ObjId) -> Result<(), InteractionError> {
        self.topology.delete(i, j)
    }

    /// Get `(k, l0)` for edge `e`.
    pub fn edge_params(&self, e: EdgeId) -> Result<(f64, f64), InteractionError> {
        if e >= self.k.len() || e >= self.l0.len() {
            return Err(InteractionError::InvalidEdgeId {
                edge: e,
                n_edges: self.k.len().min(self.l0.len()),
            });
        }
        Ok((self.k[e], self.l0[e]))
    }

    /// Update `(k, l0)` for edge `e`.
    pub fn set_edge_params(&mut self, e: EdgeId, k: f64, l0: f64) -> Result<(), InteractionError> {
        Self::validate_param("k", k, true)?;
        Self::validate_param("l0", l0, false)?;

        if e >= self.k.len() || e >= self.l0.len() {
            return Err(InteractionError::InvalidEdgeId {
                edge: e,
                n_edges: self.k.len().min(self.l0.len()),
            });
        }
        self.k[e] = k;
        self.l0[e] = l0;
        Ok(())
    }

    #[inline]
    pub fn topology(&self) -> &PairTopology {
        &self.topology
    }

    #[inline]
    pub fn topology_mut(&mut self) -> &mut PairTopology {
        &mut self.topology
    }

    #[inline]
    fn validate_param(field: &'static str, value: f64, strictly_positive: bool) -> Result<(), InteractionError> {
        if !value.is_finite() {
            return Err(InteractionError::InvalidParam { field, value });
        }
        if strictly_positive && value <= 0.0 {
            return Err(InteractionError::InvalidParam { field, value });
        }
        if !strictly_positive && value < 0.0 {
            return Err(InteractionError::InvalidParam { field, value });
        }
        Ok(())
    }

    #[inline]
    fn check_param_lengths(&self) -> Result<(), InteractionError> {
        let n_edges = self.topology.len_edges();
        if self.k.len() != n_edges || self.l0.len() != n_edges {
            return Err(InteractionError::ParamLengthMismatch {
                n_edges,
                n_k: self.k.len(),
                n_l0: self.l0.len(),
            });
        }
        Ok(())
    }
}

impl<const D: usize> Interaction<D> for SpringInteraction {
    #[inline]
    fn topology(&self) -> &PairTopology {
        &self.topology
    }

    #[inline]
    fn topology_mut(&mut self) -> &mut PairTopology {
        &mut self.topology
    }

    /**
    Accumulate Hooke accelerations over all active edges:
    `F = -k * (|r_j-r_i| - l0) * u_ij`, then `a_i += F * inv_mass_i`, `a_j -= F * inv_mass_j`.
    */
    fn accumulate_acc(&self, objects: &mut PhysObj<D>) -> Result<(), InteractionError> {
        self.check_param_lengths()?;

        for &e in self.topology.active_edges() {
            let (i, j) = self.topology.edge_pair(e)?;

            // Skip dead objects in soft-lifecycle mode.
            if !objects.is_alive(i)? || !objects.is_alive(j)? {
                continue;
            }

            // Build displacement vector and norm.
            let mut u = [0.0f64; D];
            let mut nsq = 0.0f64;
            {
                let ri = objects.pos_of(i)?;
                let rj = objects.pos_of(j)?;
                for d in 0..D {
                    let dr = rj[d] - ri[d];
                    u[d] = dr;
                    nsq += dr * dr;
                }
            }

            let norm = nsq.sqrt();
            if !norm.is_finite() || norm == 0.0 {
                continue;
            }
            let inv_norm = 1.0 / norm;
            for x in &mut u {
                *x *= inv_norm;
            }

            let f_mag = -self.k[e] * (norm - self.l0[e]);
            let ai_scale = f_mag * objects.inv_mass_of(i)?;
            let aj_scale = f_mag * objects.inv_mass_of(j)?;

            {
                let ai = objects.acc_of_mut(i)?;
                for d in 0..D {
                    ai[d] += ai_scale * u[d];
                }
            }
            {
                let aj = objects.acc_of_mut(j)?;
                for d in 0..D {
                    aj[d] -= aj_scale * u[d];
                }
            }
        }
        Ok(())
    }
}

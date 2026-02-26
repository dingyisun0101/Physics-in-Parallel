/*!
Spring-network interaction wrapper for particle models.
*/

use crate::engines::soa::interaction::DirectionMode;
use crate::engines::soa::{EdgeId, Interaction, InteractionError};

/// Optional distance window `(min, max)` where this spring is active.
pub type SpringCutoff = (f64, f64);

/// Per-edge spring payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spring {
    /// Spring constant.
    pub k: f64,
    /// Unstretched length.
    pub l_0: f64,
    /// Optional pair-distance cutoff `(min, max)`.
    pub cutoff: Option<SpringCutoff>,
}

/// Undirected network of pairwise springs.
#[derive(Debug, Clone)]
pub struct SpringNetwork {
    springs: Interaction<Spring>,
}

impl Default for SpringNetwork {
    fn default() -> Self {
        Self::empty()
    }
}

impl SpringNetwork {
    /// Creates an empty spring network.
    pub fn empty() -> Self {
        Self {
            springs: Interaction::new(0, DirectionMode::Undirected),
        }
    }

    /// Number of active springs.
    pub fn len(&self) -> usize {
        self.springs.topology().active_count()
    }

    /// Returns true if the network has no springs.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds or overwrites spring parameters for an undirected particle pair.
    pub fn add_spring(
        &mut self,
        pair: (usize, usize),
        k: f64,
        l_0: f64,
        cutoff: Option<SpringCutoff>,
    ) -> Result<EdgeId, InteractionError> {
        self.add_spring_payload(pair, Spring { k, l_0, cutoff })
    }

    /// Adds or overwrites one spring payload for an undirected particle pair.
    pub fn add_spring_payload(
        &mut self,
        pair: (usize, usize),
        spring: Spring,
    ) -> Result<EdgeId, InteractionError> {
        self.ensure_n_objects_for(pair);
        self.springs.insert(&[pair.0, pair.1], spring)
    }

    /// Removes one spring by particle pair.
    pub fn remove_spring(&mut self, pair: (usize, usize)) -> Result<Option<Spring>, InteractionError> {
        if pair.0.max(pair.1) >= self.springs.topology().n_objects() {
            return Ok(None);
        }
        Ok(self.springs.remove(&[pair.0, pair.1])?.map(|(_, spring)| spring))
    }

    /// Returns an immutable spring payload by particle pair.
    pub fn get_spring(&self, pair: (usize, usize)) -> Result<Option<&Spring>, InteractionError> {
        if pair.0.max(pair.1) >= self.springs.topology().n_objects() {
            return Ok(None);
        }
        self.springs.get(&[pair.0, pair.1])
    }

    /// Returns a mutable spring payload by particle pair.
    pub fn get_spring_mut(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<&mut Spring>, InteractionError> {
        if pair.0.max(pair.1) >= self.springs.topology().n_objects() {
            return Ok(None);
        }
        self.springs.get_mut(&[pair.0, pair.1])
    }

    /// Clears all springs while preserving allocated capacity.
    pub fn clear(&mut self) {
        self.springs.clear();
    }

    /// Read-only access to the wrapped interaction backend.
    pub fn interaction(&self) -> &Interaction<Spring> {
        &self.springs
    }

    /// Mutable access to the wrapped interaction backend.
    pub fn interaction_mut(&mut self) -> &mut Interaction<Spring> {
        &mut self.springs
    }

    /// Parallel read-only visit over active springs as `(i, j, spring)` tuples.
    pub fn par_iter_springs<F>(&self, f: F)
    where
        F: Fn(usize, usize, &Spring) + Send + Sync,
    {
        self.springs.par_for_each_active(|_edge, key, spring| {
            debug_assert_eq!(
                key.nodes.len(),
                2,
                "SpringNetwork expects pairwise edges (arity=2)"
            );

            if key.nodes.len() == 2 {
                f(key.nodes[0], key.nodes[1], spring);
            }
        });
    }

    fn ensure_n_objects_for(&mut self, pair: (usize, usize)) {
        let needed = pair.0.max(pair.1).saturating_add(1);
        if needed > self.springs.topology().n_objects() {
            self.springs.topology_mut().set_n_objects(needed);
        }
    }
}

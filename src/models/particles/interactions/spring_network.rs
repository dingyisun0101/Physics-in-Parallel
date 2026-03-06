/*!
Spring-network interaction wrapper for particle models.
*/

use crate::engines::soa::PhysObj;
use crate::engines::soa::interaction::DirectionMode;
use crate::engines::soa::{EdgeId, Interaction, InteractionError};
use crate::models::particles::attrs::{ATTR_A, ATTR_ALIVE, ATTR_M_INV, ATTR_R, ATTR_RIGID};

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

    /// Applies Hooke-law acceleration contributions for all active springs.
    ///
    /// Semantics:
    /// - For rigid/non-rigid pairs, the spring is still evaluated and only the non-rigid endpoint is updated.
    /// - For rigid/rigid pairs, no acceleration is written.
    /// - If `include_dead` is `false` and `alive` exists, dead endpoints are skipped.
    pub fn apply_hooke_acceleration(
        &self,
        objects: &mut PhysObj,
        include_dead: bool,
    ) -> Result<(), InteractionError> {
        let (dim, n, r_data, m_inv_data, alive_flags, rigid_flags) = {
            let r = objects.core.get::<f64>(ATTR_R)?;
            let m_inv = objects.core.get::<f64>(ATTR_M_INV)?;

            if r.dim() == 0 || r.num_vectors() == 0 {
                return Ok(());
            }
            if m_inv.dim() != 1 || m_inv.num_vectors() != r.num_vectors() {
                return Ok(());
            }

            let dim = r.dim();
            let n = r.num_vectors();

            let mut r_data = vec![0.0f64; n * dim];
            for i in 0..n {
                for k in 0..dim {
                    r_data[i * dim + k] = r.get(i as isize, k as isize);
                }
            }

            let mut m_inv_data = vec![0.0f64; n];
            for i in 0..n {
                m_inv_data[i] = m_inv.get(i as isize, 0);
            }

            let alive_flags = if !include_dead && objects.core.contains(ATTR_ALIVE) {
                let alive = objects.core.get::<f64>(ATTR_ALIVE)?;
                if alive.dim() != 1 || alive.num_vectors() != n {
                    None
                } else {
                    Some(
                        (0..n)
                            .map(|i| alive.get(i as isize, 0) > 0.0)
                            .collect::<Vec<bool>>(),
                    )
                }
            } else {
                None
            };

            let rigid_flags = if objects.core.contains(ATTR_RIGID) {
                let rigid = objects.core.get::<f64>(ATTR_RIGID)?;
                if rigid.dim() != 1 || rigid.num_vectors() != n {
                    None
                } else {
                    Some(
                        (0..n)
                            .map(|i| rigid.get(i as isize, 0) > 0.0)
                            .collect::<Vec<bool>>(),
                    )
                }
            } else {
                None
            };

            (dim, n, r_data, m_inv_data, alive_flags, rigid_flags)
        };

        let mut accum = vec![0.0f64; n * dim];
        let mut dr = vec![0.0f64; dim];

        for (_edge, key, spring) in self.springs.iter_active() {
            if key.nodes.len() != 2 {
                continue;
            }
            let i = key.nodes[0];
            let j = key.nodes[1];
            if i >= n || j >= n || i == j {
                continue;
            }

            if let Some(alive) = &alive_flags {
                if !alive[i] || !alive[j] {
                    continue;
                }
            }

            for k in 0..dim {
                dr[k] = r_data[i * dim + k] - r_data[j * dim + k];
            }
            let norm_sq = dr.iter().map(|x| x * x).sum::<f64>();
            if !norm_sq.is_finite() || norm_sq <= f64::EPSILON {
                continue;
            }
            let norm = norm_sq.sqrt();

            if let Some((cut_min, cut_max)) = spring.cutoff {
                if norm < cut_min || norm > cut_max {
                    continue;
                }
            }

            let f_mag = -spring.k * (norm - spring.l_0);
            let i_rigid = rigid_flags.as_ref().is_some_and(|flags| flags[i]);
            let j_rigid = rigid_flags.as_ref().is_some_and(|flags| flags[j]);

            for k in 0..dim {
                let force = f_mag * (dr[k] / norm);
                if !i_rigid {
                    accum[i * dim + k] += force * m_inv_data[i];
                }
                if !j_rigid {
                    accum[j * dim + k] -= force * m_inv_data[j];
                }
            }
        }

        let a = objects.core.get_mut::<f64>(ATTR_A)?;
        if a.dim() != dim || a.num_vectors() != n {
            return Ok(());
        }

        for i in 0..n {
            for k in 0..dim {
                let old = a.get(i as isize, k as isize);
                a.set(i as isize, k as isize, old + accum[i * dim + k]);
            }
        }
        Ok(())
    }

    fn ensure_n_objects_for(&mut self, pair: (usize, usize)) {
        let needed = pair.0.max(pair.1).saturating_add(1);
        if needed > self.springs.topology().n_objects() {
            self.springs.topology_mut().set_n_objects(needed);
        }
    }
}

/*!
Pairwise Hooke-law spring interactions for massive-particle models.

Purpose:
`SpringNetwork` stores an unordered set of particle pairs, with one `Spring`
payload on each pair. It can then add Hooke-law acceleration contributions into
the canonical particle acceleration attribute `ATTR_A`.

Design:
The network owns only interaction topology and spring parameters. Particle
state lives in `PhysObj`; `apply_hooke_acceleration` reads `ATTR_R`,
`ATTR_M_INV`, optional `ATTR_ALIVE`, optional `ATTR_RIGID`, and adds into
`ATTR_A`. Existing acceleration is preserved and spring contributions are added
on top of it.
*/

use crate::engines::soa::interaction::InteractionOrder;
use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::engines::soa::{Interaction, InteractionError, InteractionId};
use crate::models::particles::attrs::{
    ATTR_A, ATTR_ALIVE, ATTR_M_INV, ATTR_R, ATTR_RIGID, ParticleSelection, is_alive_value,
};

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

impl Spring {
    /// Builds a validated spring payload.
    pub fn new(k: f64, l_0: f64, cutoff: Option<SpringCutoff>) -> Result<Self, SpringNetworkError> {
        let spring = Self { k, l_0, cutoff };
        spring.validate()?;
        Ok(spring)
    }

    /// Validates physical spring parameters.
    pub fn validate(&self) -> Result<(), SpringNetworkError> {
        if !self.k.is_finite() {
            return Err(SpringNetworkError::InvalidSpringConstant { k: self.k });
        }
        if !self.l_0.is_finite() || self.l_0 < 0.0 {
            return Err(SpringNetworkError::InvalidRestLength { l_0: self.l_0 });
        }
        if let Some((min, max)) = self.cutoff {
            if !min.is_finite() || !max.is_finite() || min < 0.0 || max < min {
                return Err(SpringNetworkError::InvalidCutoff { min, max });
            }
        }
        Ok(())
    }
}

/// Errors returned by spring-network operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SpringNetworkError {
    /// Lower-level attribute/core access error.
    Attrs(AttrsError),
    /// Lower-level interaction storage error.
    Interaction(InteractionError),
    /// Spring constant is not finite.
    InvalidSpringConstant {
        /// Invalid spring constant.
        k: f64,
    },
    /// Rest length is not finite or is negative.
    InvalidRestLength {
        /// Invalid rest length.
        l_0: f64,
    },
    /// Cutoff bounds are not finite, negative, or ordered incorrectly.
    InvalidCutoff {
        /// Lower active distance.
        min: f64,
        /// Upper active distance.
        max: f64,
    },
    /// Required particle attribute has the wrong vector dimension.
    InvalidAttrShape {
        /// Attribute label that failed validation.
        label: &'static str,
        /// Expected vector dimension.
        expected_dim: usize,
        /// Actual vector dimension.
        got_dim: usize,
    },
    /// Attribute row count does not match the position row count.
    InconsistentParticleCount {
        /// Attribute label that failed validation.
        label: &'static str,
        /// Expected number of particle rows.
        expected: usize,
        /// Actual number of rows.
        got: usize,
    },
    /// Inverse mass is not finite or is negative.
    InvalidInverseMass {
        /// Particle row index.
        index: usize,
        /// Invalid inverse mass value.
        value: f64,
    },
    /// Internal interaction storage contained a non-pair entry.
    InvalidSpringArity {
        /// Interaction id with the wrong arity.
        id: InteractionId,
        /// Actual number of nodes.
        arity: usize,
    },
}

impl From<InteractionError> for SpringNetworkError {
    fn from(value: InteractionError) -> Self {
        Self::Interaction(value)
    }
}

impl From<AttrsError> for SpringNetworkError {
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
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
            springs: Interaction::new(0, InteractionOrder::Unordered),
        }
    }

    /// Number of active springs.
    pub fn len(&self) -> usize {
        self.springs.len()
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
    ) -> Result<InteractionId, SpringNetworkError> {
        self.add_spring_payload(pair, Spring::new(k, l_0, cutoff)?)
    }

    /// Adds or overwrites one spring payload for an undirected particle pair.
    pub fn add_spring_payload(
        &mut self,
        pair: (usize, usize),
        spring: Spring,
    ) -> Result<InteractionId, SpringNetworkError> {
        spring.validate()?;
        self.ensure_n_objects_for(pair);
        Ok(self.springs.set_pair(pair.0, pair.1, spring)?)
    }

    /// Removes one spring by particle pair.
    pub fn remove_spring(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<Spring>, SpringNetworkError> {
        if pair.0.max(pair.1) >= self.springs.topology().n_objects() {
            return Ok(None);
        }
        Ok(self
            .springs
            .remove_pair(pair.0, pair.1)?
            .map(|(_, spring)| spring))
    }

    /// Returns an immutable spring payload by particle pair.
    pub fn get_spring(&self, pair: (usize, usize)) -> Result<Option<&Spring>, SpringNetworkError> {
        if pair.0.max(pair.1) >= self.springs.topology().n_objects() {
            return Ok(None);
        }
        Ok(self.springs.get_pair(pair.0, pair.1)?)
    }

    /// Returns a mutable spring payload by particle pair.
    pub fn get_spring_mut(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<&mut Spring>, SpringNetworkError> {
        if pair.0.max(pair.1) >= self.springs.topology().n_objects() {
            return Ok(None);
        }
        Ok(self.springs.get_pair_mut(pair.0, pair.1)?)
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
        self.springs.par_for_each(|_id, nodes, spring| {
            debug_assert_eq!(
                nodes.nodes.len(),
                2,
                "SpringNetwork expects pairwise interactions (arity=2)"
            );

            if nodes.nodes.len() == 2 {
                f(nodes.nodes[0], nodes.nodes[1], spring);
            }
        });
    }

    /// Applies Hooke-law acceleration contributions for all active springs.
    ///
    /// Semantics:
    /// - For rigid/non-rigid pairs, the spring is still evaluated and only the non-rigid endpoint is updated.
    /// - For rigid/rigid pairs, no acceleration is written.
    /// - With `ParticleSelection::AliveOnly`, springs touching dead particles are skipped.
    /// - Use `ParticleSelection::All` only when intentionally debugging all allocated slots.
    pub fn apply_hooke_acceleration(
        &self,
        objects: &mut PhysObj,
        selection: ParticleSelection,
    ) -> Result<(), SpringNetworkError> {
        let (dim, n, r_data, m_inv_data, alive_flags, rigid_flags) = {
            let r = objects.core.get::<f64>(ATTR_R)?;
            let m_inv = objects.core.get::<f64>(ATTR_M_INV)?;

            if r.dim() == 0 || r.num_vectors() == 0 {
                return Ok(());
            }
            if m_inv.dim() != 1 || m_inv.num_vectors() != r.num_vectors() {
                return Err(invalid_attr_or_count(
                    ATTR_M_INV,
                    1,
                    m_inv.dim(),
                    r.num_vectors(),
                    m_inv.num_vectors(),
                ));
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
                if !m_inv_data[i].is_finite() || m_inv_data[i] < 0.0 {
                    return Err(SpringNetworkError::InvalidInverseMass {
                        index: i,
                        value: m_inv_data[i],
                    });
                }
            }

            let alive_flags = if !selection.includes_dead() && objects.core.contains(ATTR_ALIVE) {
                let alive = objects.core.get::<u8>(ATTR_ALIVE)?;
                validate_scalar_mask(ATTR_ALIVE, alive.dim(), alive.num_vectors(), n)?;
                Some(
                    (0..n)
                        .map(|i| is_alive_value(alive.get(i as isize, 0)))
                        .collect::<Vec<bool>>(),
                )
            } else {
                None
            };

            let rigid_flags = if objects.core.contains(ATTR_RIGID) {
                let rigid = objects.core.get::<f64>(ATTR_RIGID)?;
                validate_scalar_mask(ATTR_RIGID, rigid.dim(), rigid.num_vectors(), n)?;
                Some(
                    (0..n)
                        .map(|i| rigid.get(i as isize, 0) > 0.0)
                        .collect::<Vec<bool>>(),
                )
            } else {
                None
            };

            (dim, n, r_data, m_inv_data, alive_flags, rigid_flags)
        };

        let mut accum = vec![0.0f64; n * dim];
        let mut dr = vec![0.0f64; dim];

        for (id, nodes, spring) in self.springs.iter() {
            if nodes.nodes.len() != 2 {
                return Err(SpringNetworkError::InvalidSpringArity {
                    id,
                    arity: nodes.nodes.len(),
                });
            }
            spring.validate()?;

            let i = nodes.nodes[0];
            let j = nodes.nodes[1];
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
            return Err(invalid_attr_or_count(
                ATTR_A,
                dim,
                a.dim(),
                n,
                a.num_vectors(),
            ));
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
            self.springs
                .set_n_objects(needed)
                .expect("growing spring interaction object bound should not invalidate entries");
        }
    }
}

fn validate_scalar_mask(
    label: &'static str,
    got_dim: usize,
    got_n: usize,
    expected_n: usize,
) -> Result<(), SpringNetworkError> {
    if got_dim != 1 {
        return Err(SpringNetworkError::InvalidAttrShape {
            label,
            expected_dim: 1,
            got_dim,
        });
    }
    if got_n != expected_n {
        return Err(SpringNetworkError::InconsistentParticleCount {
            label,
            expected: expected_n,
            got: got_n,
        });
    }
    Ok(())
}

fn invalid_attr_or_count(
    label: &'static str,
    expected_dim: usize,
    got_dim: usize,
    expected_n: usize,
    got_n: usize,
) -> SpringNetworkError {
    if got_dim != expected_dim {
        SpringNetworkError::InvalidAttrShape {
            label,
            expected_dim,
            got_dim,
        }
    } else {
        SpringNetworkError::InconsistentParticleCount {
            label,
            expected: expected_n,
            got: got_n,
        }
    }
}

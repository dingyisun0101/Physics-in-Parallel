/*!
Pairwise Hooke-law spring interactions for massive-particle models.

Purpose:
`SpringNetwork` stores an unordered set of particle pairs, with one reusable
`models::laws::Spring` payload on each pair. It can then add Hooke-law
acceleration contributions into the canonical particle acceleration attribute
`ATTR_A`.

Design:
The network owns only interaction topology and spring parameters. Particle
state lives in `PhysObj`; `apply_hooke_acceleration` reads `ATTR_R`,
`ATTR_M_INV`, optional `ATTR_ALIVE`, optional `ATTR_RIGID`, and adds into
`ATTR_A`. Existing acceleration is preserved and spring contributions are added
on top of it.
*/

use core::fmt;

use crate::engines::soa::interaction::InteractionOrder;
use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::engines::soa::{Interaction, InteractionError, InteractionId};
use crate::models::laws::{Spring, SpringCutoff, SpringLawError};
use crate::models::particles::attrs::{ATTR_A, ATTR_R, ParticleSelection};
use crate::models::particles::state::{ParticleStateError, gather_inverse_mass, gather_masks};

/// Errors returned by spring-network operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SpringNetworkError {
    /// Lower-level attribute/core access error.
    Attrs(AttrsError),
    /// Lower-level interaction storage error.
    Interaction(InteractionError),
    /// Lower-level spring law validation error.
    Law(SpringLawError),
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

impl From<SpringLawError> for SpringNetworkError {
    fn from(value: SpringLawError) -> Self {
        Self::Law(value)
    }
}

impl From<ParticleStateError> for SpringNetworkError {
    fn from(value: ParticleStateError) -> Self {
        match value {
            ParticleStateError::Attrs(err) => Self::Attrs(err),
            ParticleStateError::InvalidAttrShape {
                label,
                expected_dim,
                got_dim,
            } => Self::InvalidAttrShape {
                label,
                expected_dim,
                got_dim,
            },
            ParticleStateError::InconsistentParticleCount {
                label,
                expected,
                got,
            } => Self::InconsistentParticleCount {
                label,
                expected,
                got,
            },
        }
    }
}

impl fmt::Display for SpringNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attrs(error) => write!(f, "spring-network attribute error: {error}"),
            Self::Interaction(error) => write!(f, "spring-network topology error: {error}"),
            Self::Law(error) => write!(f, "spring-network law error: {error}"),
            Self::InvalidAttrShape {
                label,
                expected_dim,
                got_dim,
            } => write!(
                f,
                "spring-network attribute `{label}` has dimension {got_dim}; expected {expected_dim}"
            ),
            Self::InconsistentParticleCount {
                label,
                expected,
                got,
            } => write!(
                f,
                "spring-network attribute `{label}` has {got} particles; expected {expected}"
            ),
            Self::InvalidInverseMass { index, value } => write!(
                f,
                "spring-network inverse mass at particle {index} must be finite and non-negative; got {value}"
            ),
            Self::InvalidSpringArity { id, arity } => {
                write!(f, "spring interaction {id} has arity {arity}; expected 2")
            }
        }
    }
}

impl std::error::Error for SpringNetworkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Attrs(error) => Some(error),
            Self::Interaction(error) => Some(error),
            Self::Law(error) => Some(error),
            _ => None,
        }
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

    /// Creates an empty spring network with a known particle bound and spring capacity.
    pub fn with_capacity(num_particles: usize, spring_capacity: usize) -> Self {
        let mut springs = Interaction::new(num_particles, InteractionOrder::Unordered);
        springs.reserve(spring_capacity);
        Self { springs }
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

    /// Adds or overwrites many springs that share one payload.
    pub fn add_springs_payload(
        &mut self,
        pairs: &[(usize, usize)],
        spring: Spring,
    ) -> Result<(), SpringNetworkError> {
        spring.validate()?;
        if let Some(max_obj) = pairs.iter().map(|&(i, j)| i.max(j)).max() {
            self.ensure_n_objects(max_obj.saturating_add(1));
        }

        for &(i, j) in pairs {
            self.springs.set_pair(i, j, spring)?;
        }
        Ok(())
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
        let (dim, n, r_data, m_inv_data, masks) = {
            let r = objects.core.get::<f64>(ATTR_R)?;

            if r.dim() == 0 || r.num_vectors() == 0 {
                return Ok(());
            }

            let dim = r.dim();
            let n = r.num_vectors();

            let r_data = r.as_tensor().data.clone();

            let m_inv_data = gather_inverse_mass(objects, n)?;
            for (i, &value) in m_inv_data.iter().enumerate().take(n) {
                if !value.is_finite() || value < 0.0 {
                    return Err(SpringNetworkError::InvalidInverseMass { index: i, value });
                }
            }

            let masks = gather_masks(objects, n, selection)?;

            (dim, n, r_data, m_inv_data, masks)
        };

        let mut accum = vec![0.0f64; n * dim];

        match dim {
            1 => accumulate_hooke_1d(
                &self.springs,
                &r_data,
                &m_inv_data,
                &masks,
                selection,
                n,
                &mut accum,
            )?,
            2 => accumulate_hooke_2d(
                &self.springs,
                &r_data,
                &m_inv_data,
                &masks,
                selection,
                n,
                &mut accum,
            )?,
            3 => accumulate_hooke_3d(
                &self.springs,
                &r_data,
                &m_inv_data,
                &masks,
                selection,
                n,
                &mut accum,
            )?,
            _ => accumulate_hooke_generic(
                &self.springs,
                &r_data,
                &m_inv_data,
                &masks,
                selection,
                n,
                dim,
                &mut accum,
            )?,
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

        for (dst, src) in a.as_tensor_mut().data.iter_mut().zip(accum) {
            *dst += src;
        }
        Ok(())
    }

    fn ensure_n_objects_for(&mut self, pair: (usize, usize)) {
        let needed = pair.0.max(pair.1).saturating_add(1);
        self.ensure_n_objects(needed);
    }

    fn ensure_n_objects(&mut self, needed: usize) {
        if needed > self.springs.topology().n_objects() {
            self.springs
                .set_n_objects(needed)
                .expect("growing spring interaction object bound should not invalidate entries");
        }
    }
}

fn accumulate_hooke_1d(
    springs: &Interaction<Spring>,
    r_data: &[f64],
    m_inv_data: &[f64],
    masks: &crate::models::particles::state::ParticleMasks,
    selection: ParticleSelection,
    n: usize,
    accum: &mut [f64],
) -> Result<(), SpringNetworkError> {
    for (id, nodes, spring) in springs.iter() {
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

        if !masks.is_included(selection, i) || !masks.is_included(selection, j) {
            continue;
        }

        let dx = r_data[i] - r_data[j];
        let norm = dx.abs();
        if !norm.is_finite() || norm <= f64::EPSILON {
            continue;
        }

        if let Some((cut_min, cut_max)) = spring.cutoff
            && (norm < cut_min || norm > cut_max)
        {
            continue;
        }

        let force = dx * (-spring.k * (norm - spring.l_0) / norm);
        let i_rigid = masks.rigid.as_ref().is_some_and(|flags| flags[i]);
        let j_rigid = masks.rigid.as_ref().is_some_and(|flags| flags[j]);

        if !i_rigid {
            accum[i] += force * m_inv_data[i];
        }
        if !j_rigid {
            accum[j] -= force * m_inv_data[j];
        }
    }

    Ok(())
}

fn accumulate_hooke_2d(
    springs: &Interaction<Spring>,
    r_data: &[f64],
    m_inv_data: &[f64],
    masks: &crate::models::particles::state::ParticleMasks,
    selection: ParticleSelection,
    n: usize,
    accum: &mut [f64],
) -> Result<(), SpringNetworkError> {
    for (id, nodes, spring) in springs.iter() {
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

        if !masks.is_included(selection, i) || !masks.is_included(selection, j) {
            continue;
        }

        let i_base = i * 2;
        let j_base = j * 2;
        let dx = r_data[i_base] - r_data[j_base];
        let dy = r_data[i_base + 1] - r_data[j_base + 1];
        let norm_sq = dx * dx + dy * dy;
        if !norm_sq.is_finite() || norm_sq <= f64::EPSILON {
            continue;
        }
        let norm = norm_sq.sqrt();

        if let Some((cut_min, cut_max)) = spring.cutoff
            && (norm < cut_min || norm > cut_max)
        {
            continue;
        }

        let scale = -spring.k * (norm - spring.l_0) / norm;
        let i_rigid = masks.rigid.as_ref().is_some_and(|flags| flags[i]);
        let j_rigid = masks.rigid.as_ref().is_some_and(|flags| flags[j]);

        if !i_rigid {
            let i_scale = scale * m_inv_data[i];
            accum[i_base] += dx * i_scale;
            accum[i_base + 1] += dy * i_scale;
        }
        if !j_rigid {
            let j_scale = scale * m_inv_data[j];
            accum[j_base] -= dx * j_scale;
            accum[j_base + 1] -= dy * j_scale;
        }
    }

    Ok(())
}

fn accumulate_hooke_3d(
    springs: &Interaction<Spring>,
    r_data: &[f64],
    m_inv_data: &[f64],
    masks: &crate::models::particles::state::ParticleMasks,
    selection: ParticleSelection,
    n: usize,
    accum: &mut [f64],
) -> Result<(), SpringNetworkError> {
    for (id, nodes, spring) in springs.iter() {
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

        if !masks.is_included(selection, i) || !masks.is_included(selection, j) {
            continue;
        }

        let i_base = i * 3;
        let j_base = j * 3;
        let dx = r_data[i_base] - r_data[j_base];
        let dy = r_data[i_base + 1] - r_data[j_base + 1];
        let dz = r_data[i_base + 2] - r_data[j_base + 2];
        let norm_sq = dx * dx + dy * dy + dz * dz;
        if !norm_sq.is_finite() || norm_sq <= f64::EPSILON {
            continue;
        }
        let norm = norm_sq.sqrt();

        if let Some((cut_min, cut_max)) = spring.cutoff
            && (norm < cut_min || norm > cut_max)
        {
            continue;
        }

        let scale = -spring.k * (norm - spring.l_0) / norm;
        let i_rigid = masks.rigid.as_ref().is_some_and(|flags| flags[i]);
        let j_rigid = masks.rigid.as_ref().is_some_and(|flags| flags[j]);

        if !i_rigid {
            let i_scale = scale * m_inv_data[i];
            accum[i_base] += dx * i_scale;
            accum[i_base + 1] += dy * i_scale;
            accum[i_base + 2] += dz * i_scale;
        }
        if !j_rigid {
            let j_scale = scale * m_inv_data[j];
            accum[j_base] -= dx * j_scale;
            accum[j_base + 1] -= dy * j_scale;
            accum[j_base + 2] -= dz * j_scale;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn accumulate_hooke_generic(
    springs: &Interaction<Spring>,
    r_data: &[f64],
    m_inv_data: &[f64],
    masks: &crate::models::particles::state::ParticleMasks,
    selection: ParticleSelection,
    n: usize,
    dim: usize,
    accum: &mut [f64],
) -> Result<(), SpringNetworkError> {
    let mut dr = vec![0.0f64; dim];

    for (id, nodes, spring) in springs.iter() {
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

        if !masks.is_included(selection, i) || !masks.is_included(selection, j) {
            continue;
        }

        for k in 0..dim {
            dr[k] = r_data[i * dim + k] - r_data[j * dim + k];
        }
        let norm_sq = dr.iter().map(|x| x * x).sum::<f64>();
        if !norm_sq.is_finite() || norm_sq <= f64::EPSILON {
            continue;
        }
        let norm = norm_sq.sqrt();

        if let Some((cut_min, cut_max)) = spring.cutoff
            && (norm < cut_min || norm > cut_max)
        {
            continue;
        }

        let scale = -spring.k * (norm - spring.l_0) / norm;
        let i_rigid = masks.rigid.as_ref().is_some_and(|flags| flags[i]);
        let j_rigid = masks.rigid.as_ref().is_some_and(|flags| flags[j]);

        for k in 0..dim {
            let dr_k = dr[k];
            if !i_rigid {
                accum[i * dim + k] += dr_k * scale * m_inv_data[i];
            }
            if !j_rigid {
                accum[j * dim + k] -= dr_k * scale * m_inv_data[j];
            }
        }
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

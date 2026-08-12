/*!
Pairwise power-law interaction parameter storage for particle models.

Purpose:
`PowerLawNetwork` stores unordered particle pairs with one reusable
`models::laws::PowerLawDecay` payload on each pair. It can then add acceleration
contributions into the canonical particle acceleration attribute `ATTR_A`.
*/

use crate::engines::soa::interaction::InteractionOrder;
use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::engines::soa::{Interaction, InteractionError, InteractionId};
use crate::models::laws::{PowerLawDecay, PowerLawError, PowerLawRange};
use crate::models::particles::attrs::{ATTR_A, ATTR_R, ParticleSelection};
use crate::models::particles::state::{ParticleStateError, gather_inverse_mass, gather_masks};

/// Errors returned by power-law network operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PowerLawNetworkError {
    /// Lower-level interaction storage error.
    Interaction(InteractionError),
    /// Lower-level attribute/core access error.
    Attrs(AttrsError),
    /// Lower-level power-law validation error.
    Law(PowerLawError),
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
    InvalidPowerLawArity {
        /// Interaction id with the wrong arity.
        id: InteractionId,
        /// Actual number of nodes.
        arity: usize,
    },
}

impl From<InteractionError> for PowerLawNetworkError {
    fn from(value: InteractionError) -> Self {
        Self::Interaction(value)
    }
}

impl From<AttrsError> for PowerLawNetworkError {
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
}

impl From<PowerLawError> for PowerLawNetworkError {
    fn from(value: PowerLawError) -> Self {
        Self::Law(value)
    }
}

impl From<ParticleStateError> for PowerLawNetworkError {
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

/// Undirected network of pairwise power-law interaction parameters.
#[derive(Debug, Clone)]
pub struct PowerLawNetwork {
    interactions: Interaction<PowerLawDecay>,
}

impl Default for PowerLawNetwork {
    fn default() -> Self {
        Self::empty()
    }
}

impl PowerLawNetwork {
    /// Creates an empty power-law network.
    pub fn empty() -> Self {
        Self {
            interactions: Interaction::new(0, InteractionOrder::Unordered),
        }
    }

    /// Creates an empty power-law network with a known particle bound and pair capacity.
    pub fn with_capacity(num_particles: usize, pair_capacity: usize) -> Self {
        let mut interactions = Interaction::new(num_particles, InteractionOrder::Unordered);
        interactions.reserve(pair_capacity);
        Self { interactions }
    }

    /// Number of unordered all-to-all pairs for `num_particles`.
    pub fn all_to_all_pair_count(num_particles: usize) -> usize {
        num_particles.saturating_mul(num_particles.saturating_sub(1)) / 2
    }

    /// Creates an empty all-to-all power-law network with enough capacity for every pair.
    pub fn all_to_all_empty(num_particles: usize) -> Self {
        Self::with_capacity(num_particles, Self::all_to_all_pair_count(num_particles))
    }

    /// Number of active pair interactions.
    pub fn len(&self) -> usize {
        self.interactions.len()
    }

    /// Returns true if no interactions exist.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds or overwrites one pair interaction payload from constants.
    pub fn add_power_law(
        &mut self,
        pair: (usize, usize),
        k: f64,
        alpha: f64,
        range: Option<PowerLawRange>,
    ) -> Result<InteractionId, PowerLawNetworkError> {
        self.add_payload(pair, PowerLawDecay::new(k, alpha, range)?)
    }

    /// Adds or overwrites one pair interaction payload.
    pub fn add_payload(
        &mut self,
        pair: (usize, usize),
        payload: PowerLawDecay,
    ) -> Result<InteractionId, PowerLawNetworkError> {
        payload.validate()?;
        self.ensure_n_objects_for(pair);
        Ok(self.interactions.set_pair(pair.0, pair.1, payload)?)
    }

    /// Adds or overwrites many pair interactions that share one payload.
    pub fn add_payloads(
        &mut self,
        pairs: &[(usize, usize)],
        payload: PowerLawDecay,
    ) -> Result<(), PowerLawNetworkError> {
        payload.validate()?;
        if let Some(max_obj) = pairs.iter().map(|&(i, j)| i.max(j)).max() {
            self.ensure_n_objects(max_obj.saturating_add(1));
        }

        for &(i, j) in pairs {
            self.interactions.set_pair(i, j, payload)?;
        }
        Ok(())
    }

    /// Adds or overwrites every unordered particle pair with one shared payload.
    pub fn add_all_to_all_payload(
        &mut self,
        num_particles: usize,
        payload: PowerLawDecay,
    ) -> Result<(), PowerLawNetworkError> {
        payload.validate()?;
        self.ensure_n_objects(num_particles);
        self.interactions
            .reserve(Self::all_to_all_pair_count(num_particles));

        for i in 0..num_particles {
            for j in (i + 1)..num_particles {
                self.interactions.set_pair(i, j, payload)?;
            }
        }
        Ok(())
    }

    /// Removes one pair interaction payload.
    pub fn remove_power_law(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<PowerLawDecay>, PowerLawNetworkError> {
        if pair.0.max(pair.1) >= self.interactions.topology().n_objects() {
            return Ok(None);
        }
        Ok(self
            .interactions
            .remove_pair(pair.0, pair.1)?
            .map(|(_, payload)| payload))
    }

    /// Returns immutable payload for one pair.
    pub fn get_power_law(
        &self,
        pair: (usize, usize),
    ) -> Result<Option<&PowerLawDecay>, PowerLawNetworkError> {
        if pair.0.max(pair.1) >= self.interactions.topology().n_objects() {
            return Ok(None);
        }
        Ok(self.interactions.get_pair(pair.0, pair.1)?)
    }

    /// Returns mutable payload for one pair.
    pub fn get_power_law_mut(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<&mut PowerLawDecay>, PowerLawNetworkError> {
        if pair.0.max(pair.1) >= self.interactions.topology().n_objects() {
            return Ok(None);
        }
        Ok(self.interactions.get_pair_mut(pair.0, pair.1)?)
    }

    /// Clears all interactions while preserving capacity.
    pub fn clear(&mut self) {
        self.interactions.clear();
    }

    /// Read-only access to the wrapped interaction backend.
    pub fn interaction(&self) -> &Interaction<PowerLawDecay> {
        &self.interactions
    }

    /// Mutable access to the wrapped interaction backend.
    pub fn interaction_mut(&mut self) -> &mut Interaction<PowerLawDecay> {
        &mut self.interactions
    }

    /// Applies pairwise power-law acceleration contributions for all active pairs.
    ///
    /// The convention is `force_magnitude = k * distance^alpha`, applied along
    /// the vector from particle `j` to particle `i`. Positive `k` is repulsive;
    /// negative `k` is attractive.
    pub fn apply_power_law_acceleration(
        &self,
        objects: &mut PhysObj,
        selection: ParticleSelection,
    ) -> Result<(), PowerLawNetworkError> {
        let (dim, n, r_data, m_inv_data, masks) = {
            let r = objects.core.get::<f64>(ATTR_R)?;

            if r.dim() == 0 || r.num_vectors() == 0 {
                return Ok(());
            }

            let dim = r.dim();
            let n = r.num_vectors();
            let r_data = r.as_tensor().data.clone();

            let m_inv_data = gather_inverse_mass(objects, n)?;
            for (i, &value) in m_inv_data.iter().enumerate() {
                if !value.is_finite() || value < 0.0 {
                    return Err(PowerLawNetworkError::InvalidInverseMass { index: i, value });
                }
            }

            let masks = gather_masks(objects, n, selection)?;
            (dim, n, r_data, m_inv_data, masks)
        };

        let mut accum = vec![0.0f64; n * dim];
        if dim == 3 {
            accumulate_power_law_3d(
                &self.interactions,
                &r_data,
                &m_inv_data,
                &masks,
                selection,
                n,
                &mut accum,
            )?;
        } else {
            accumulate_power_law_generic(
                &self.interactions,
                &r_data,
                &m_inv_data,
                &masks,
                selection,
                n,
                dim,
                &mut accum,
            )?;
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
        if needed > self.interactions.topology().n_objects() {
            self.interactions
                .set_n_objects(needed)
                .expect("growing power-law interaction object bound should not invalidate entries");
        }
    }
}

fn accumulate_power_law_3d(
    interactions: &Interaction<PowerLawDecay>,
    r_data: &[f64],
    m_inv_data: &[f64],
    masks: &crate::models::particles::state::ParticleMasks,
    selection: ParticleSelection,
    n: usize,
    accum: &mut [f64],
) -> Result<(), PowerLawNetworkError> {
    for (id, nodes, law) in interactions.iter() {
        if nodes.nodes.len() != 2 {
            return Err(PowerLawNetworkError::InvalidPowerLawArity {
                id,
                arity: nodes.nodes.len(),
            });
        }
        law.validate()?;

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

        if let Some((min, max)) = law.range
            && (norm < min || norm > max)
        {
            continue;
        }

        let scale = law.k * norm.powf(law.alpha - 1.0);
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
fn accumulate_power_law_generic(
    interactions: &Interaction<PowerLawDecay>,
    r_data: &[f64],
    m_inv_data: &[f64],
    masks: &crate::models::particles::state::ParticleMasks,
    selection: ParticleSelection,
    n: usize,
    dim: usize,
    accum: &mut [f64],
) -> Result<(), PowerLawNetworkError> {
    let mut dr = vec![0.0f64; dim];

    for (id, nodes, law) in interactions.iter() {
        if nodes.nodes.len() != 2 {
            return Err(PowerLawNetworkError::InvalidPowerLawArity {
                id,
                arity: nodes.nodes.len(),
            });
        }
        law.validate()?;

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

        if let Some((min, max)) = law.range
            && (norm < min || norm > max)
        {
            continue;
        }

        let scale = law.k * norm.powf(law.alpha - 1.0);
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
) -> PowerLawNetworkError {
    if got_dim != expected_dim {
        PowerLawNetworkError::InvalidAttrShape {
            label,
            expected_dim,
            got_dim,
        }
    } else {
        PowerLawNetworkError::InconsistentParticleCount {
            label,
            expected: expected_n,
            got: got_n,
        }
    }
}

/*!
Particle-facing wrapper around the SoA cell-linked neighbor list.

Purpose:
`ParticleNeighborList` connects the engine-level `NeighborList` to canonical
particle attributes. It reads positions from `ATTR_R`, rebuilds nearby-cell
candidate buckets, then filters those candidates by Euclidean distance and
optional alive-mask state.

Design:
The wrapped engine list only knows about geometry and cell buckets. This wrapper
is where particle-specific rules belong: position attribute validation, cutoff
distance checks, and optional exclusion of dead particles.
*/

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::engines::soa::{NeighborList, NeighborListError};
use crate::models::particles::attrs::{ATTR_R, ParticleSelection};
use crate::models::particles::state::{ParticleStateError, gather_alive_flags};

/// Errors returned by particle-level neighbor-list operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ParticleNeighborListError {
    /// Lower-level attribute/core access error.
    Attrs(AttrsError),
    /// Lower-level engine neighbor-list error.
    Neighbor(NeighborListError),
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
}

impl From<AttrsError> for ParticleNeighborListError {
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
}

impl From<NeighborListError> for ParticleNeighborListError {
    fn from(value: NeighborListError) -> Self {
        Self::Neighbor(value)
    }
}

impl From<ParticleStateError> for ParticleNeighborListError {
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

/// Cutoff-filtered neighbor helper for canonical massive-particle objects.
#[derive(Debug, Clone)]
pub struct ParticleNeighborList {
    /// Engine-level nearby-cell candidate generator.
    candidates: NeighborList,
    /// Physical cutoff distance used after candidate generation.
    cutoff: f64,
}

impl ParticleNeighborList {
    /// Builds a particle neighbor list over explicit rectangular bounds.
    pub fn from_bounds(
        min: &[f64],
        max: &[f64],
        cutoff: f64,
    ) -> Result<Self, ParticleNeighborListError> {
        let candidates = NeighborList::new(min, max, cutoff)?;
        Ok(Self { candidates, cutoff })
    }

    /// Builds a particle neighbor list over `[0, dimensions[k]]` on each axis.
    pub fn from_box(dimensions: &[f64], cutoff: f64) -> Result<Self, ParticleNeighborListError> {
        let min = vec![0.0f64; dimensions.len()];
        Self::from_bounds(min.as_slice(), dimensions, cutoff)
    }

    /// Returns spatial dimension.
    pub fn dim(&self) -> usize {
        self.candidates.dim()
    }

    /// Returns the physical cutoff distance.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Returns the wrapped engine-level candidate list.
    pub fn candidates(&self) -> &NeighborList {
        &self.candidates
    }

    /// Returns mutable access to the wrapped engine-level candidate list.
    pub fn candidates_mut(&mut self) -> &mut NeighborList {
        &mut self.candidates
    }

    /// Rebuilds nearby-cell candidates from `objects.core[ATTR_R]`.
    pub fn rebuild(&mut self, objects: &PhysObj) -> Result<(), ParticleNeighborListError> {
        let r = objects.core.get::<f64>(ATTR_R)?;
        self.validate_position_dim(r.dim())?;
        self.candidates
            .rebuild(r.as_tensor().data.as_slice(), r.num_vectors())?;
        Ok(())
    }

    /// Collects physical neighbor pairs within cutoff.
    ///
    /// The result contains unique unordered pairs `(i, j)` with `i < j`. If
    /// With `ParticleSelection::AliveOnly`, pairs touching dead particles are
    /// skipped. Use `ParticleSelection::All` only when intentionally debugging or
    /// inspecting all allocated slots.
    pub fn collect_pairs(
        &self,
        objects: &PhysObj,
        selection: ParticleSelection,
    ) -> Result<Vec<(usize, usize)>, ParticleNeighborListError> {
        let r = objects.core.get::<f64>(ATTR_R)?;
        self.validate_position_dim(r.dim())?;
        let dim = r.dim();
        let n = r.num_vectors();
        let r_data = r.as_tensor().data.as_slice();

        let alive_flags = gather_alive_flags(objects, n, selection)?;
        let cutoff_sq = self.cutoff * self.cutoff;
        let mut pairs = Vec::<(usize, usize)>::new();

        self.candidates.for_each_pair_candidate(|i, j| {
            if i >= n || j >= n {
                return;
            }

            if let Some(flags) = &alive_flags
                && (!flags[i] || !flags[j])
            {
                return;
            }

            let i0 = i * dim;
            let j0 = j * dim;
            let mut nsq = 0.0f64;
            for axis in 0..dim {
                let dr = r_data[j0 + axis] - r_data[i0 + axis];
                nsq += dr * dr;
            }
            if nsq.is_finite() && nsq > 0.0 && nsq < cutoff_sq {
                pairs.push((i, j));
            }
        });

        Ok(pairs)
    }

    /// Explicit alias for `collect_pairs`.
    pub fn collect_pairs_within_cutoff(
        &self,
        objects: &PhysObj,
        selection: ParticleSelection,
    ) -> Result<Vec<(usize, usize)>, ParticleNeighborListError> {
        self.collect_pairs(objects, selection)
    }

    fn validate_position_dim(&self, got_dim: usize) -> Result<(), ParticleNeighborListError> {
        let expected_dim = self.dim();
        if got_dim != expected_dim {
            return Err(ParticleNeighborListError::InvalidAttrShape {
                label: ATTR_R,
                expected_dim,
                got_dim,
            });
        }
        Ok(())
    }
}

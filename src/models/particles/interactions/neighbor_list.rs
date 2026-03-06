/*!
Particle-model wrapper around SoA cell-linked neighbor lists.
*/

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::engines::soa::{NeighborList, NeighborListError};
use crate::models::particles::attrs::{ATTR_ALIVE, ATTR_R};

#[derive(Debug, Clone, PartialEq)]
pub enum ParticleNeighborListError {
    Attrs(AttrsError),
    Neighbor(NeighborListError),
    InvalidAttrShape {
        label: &'static str,
        expected_dim: usize,
        got_dim: usize,
    },
    InconsistentParticleCount {
        label: &'static str,
        expected: usize,
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

#[derive(Debug, Clone)]
pub struct ParticleNeighborList {
    grid: NeighborList,
    cutoff: f64,
}

impl ParticleNeighborList {
    pub fn from_bounds(
        min: &[f64],
        max: &[f64],
        cutoff: f64,
    ) -> Result<Self, ParticleNeighborListError> {
        let grid = NeighborList::new(min, max, cutoff)?;
        Ok(Self { grid, cutoff })
    }

    pub fn from_box(dimensions: &[f64], cutoff: f64) -> Result<Self, ParticleNeighborListError> {
        let min = vec![0.0f64; dimensions.len()];
        Self::from_bounds(min.as_slice(), dimensions, cutoff)
    }

    pub fn rebuild(&mut self, objects: &PhysObj) -> Result<(), ParticleNeighborListError> {
        let r = objects.core.get::<f64>(ATTR_R)?;
        self.grid.rebuild(r.as_tensor().data.as_slice(), r.num_vectors())?;
        Ok(())
    }

    pub fn collect_pairs_within_cutoff(
        &self,
        objects: &PhysObj,
        include_dead: bool,
    ) -> Result<Vec<(usize, usize)>, ParticleNeighborListError> {
        let r = objects.core.get::<f64>(ATTR_R)?;
        let dim = r.dim();
        let n = r.num_vectors();
        let r_data = r.as_tensor().data.as_slice();

        let alive_flags = if include_dead || !objects.core.contains(ATTR_ALIVE) {
            None
        } else {
            let alive = objects.core.get::<f64>(ATTR_ALIVE)?;
            if alive.dim() != 1 {
                return Err(ParticleNeighborListError::InvalidAttrShape {
                    label: ATTR_ALIVE,
                    expected_dim: 1,
                    got_dim: alive.dim(),
                });
            }
            if alive.num_vectors() != n {
                return Err(ParticleNeighborListError::InconsistentParticleCount {
                    label: ATTR_ALIVE,
                    expected: n,
                    got: alive.num_vectors(),
                });
            }
            Some(
                (0..n)
                    .map(|i| alive.get(i as isize, 0) > 0.0)
                    .collect::<Vec<bool>>(),
            )
        };

        let cutoff_sq = self.cutoff * self.cutoff;
        let mut pairs = Vec::<(usize, usize)>::new();
        self.grid.for_each_candidate_pair(|i, j| {
            if let Some(flags) = &alive_flags {
                if !flags[i] || !flags[j] {
                    return;
                }
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
}

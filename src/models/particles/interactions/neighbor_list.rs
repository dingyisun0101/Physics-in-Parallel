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

use core::fmt;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::engines::soa::{NeighborList, NeighborListError};
use crate::models::particles::attrs::{ATTR_R, ParticleSelection};
use crate::models::particles::state::{BorrowedMasks, ParticleStateError};

/// Errors returned by particle-level neighbor-list operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ParticleNeighborListError {
    State(ParticleStateError),
    Geometry { message: String },
}

impl From<AttrsError> for ParticleNeighborListError {
    fn from(value: AttrsError) -> Self {
        Self::State(value.into())
    }
}

impl From<NeighborListError> for ParticleNeighborListError {
    fn from(value: NeighborListError) -> Self {
        Self::Geometry {
            message: value.to_string(),
        }
    }
}

impl From<ParticleStateError> for ParticleNeighborListError {
    fn from(value: ParticleStateError) -> Self {
        Self::State(value)
    }
}

impl fmt::Display for ParticleNeighborListError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(error) => write!(f, "invalid particle state: {error}"),
            Self::Geometry { message } => {
                write!(f, "particle neighbor-list geometry error: {message}")
            }
        }
    }
}

impl std::error::Error for ParticleNeighborListError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(error) => Some(error),
            Self::Geometry { .. } => None,
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
    built_for: Option<usize>,
}

impl ParticleNeighborList {
    /// Builds a particle neighbor list over explicit rectangular bounds.
    pub fn from_bounds(
        min: &[f64],
        max: &[f64],
        cutoff: f64,
    ) -> Result<Self, ParticleNeighborListError> {
        let candidates = NeighborList::new(min, max, cutoff)?;
        Ok(Self {
            candidates,
            cutoff,
            built_for: None,
        })
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

    /// Rebuilds nearby-cell candidates from `objects.core[ATTR_R]`.
    pub fn rebuild(&mut self, objects: &PhysObj) -> Result<(), ParticleNeighborListError> {
        let r = objects.core.get::<f64>(ATTR_R)?;
        self.validate_position_dim(r.dim())?;
        self.candidates
            .rebuild(&r.borrow_values(), r.num_vectors())?;
        self.built_for = Some(r.num_vectors());
        Ok(())
    }

    /// Collects physical neighbor pairs within cutoff.
    ///
    /// Requires unchanged positions since rebuild; prefer rebuild_and_collect.
    /// The result contains unique unordered pairs `(i, j)` with `i < j`.
    /// With `ParticleSelection::AliveOnly`, pairs touching dead particles are
    /// skipped. Use `ParticleSelection::All` only when intentionally debugging or
    /// inspecting all allocated slots.
    pub fn collect_pairs(
        &self,
        objects: &PhysObj,
        selection: ParticleSelection,
    ) -> Result<Vec<(usize, usize)>, ParticleNeighborListError> {
        let mut pairs = Vec::new();
        self.for_each_pair(objects, selection, |i, j| pairs.push((i, j)))?;
        Ok(pairs)
    }

    /// Rebuilds and queries current positions, preventing stale candidate use.
    /// Distance is strictly between zero and cutoff, with no periodic minimum
    /// image. Positions may be clipped into cells, but distance uses raw values.
    pub fn rebuild_and_collect(
        &mut self,
        objects: &PhysObj,
        selection: ParticleSelection,
    ) -> Result<Vec<(usize, usize)>, ParticleNeighborListError> {
        self.rebuild(objects)?;
        self.collect_pairs(objects, selection)
    }

    /// Rebuilds current positions and replaces a caller-owned pair buffer.
    /// Retains buffer capacity. Geometry/state errors leave output unchanged.
    pub fn rebuild_and_collect_into(
        &mut self,
        objects: &PhysObj,
        selection: ParticleSelection,
        output: &mut Vec<(usize, usize)>,
    ) -> Result<(), ParticleNeighborListError> {
        self.rebuild(objects)?;
        self.collect_pairs_into(objects, selection, output)
    }

    /// Queries a previously rebuilt list into a reusable output buffer.
    /// Positions and particle count must remain unchanged since rebuild; alive
    /// flags are read at query time. Rebuild-and-query is the normal safe path.
    /// Shape and mask errors are validated before replacing the output.
    pub fn collect_pairs_into(
        &self,
        objects: &PhysObj,
        selection: ParticleSelection,
        output: &mut Vec<(usize, usize)>,
    ) -> Result<(), ParticleNeighborListError> {
        let mut first = true;
        self.for_each_pair(objects, selection, |i, j| {
            if first {
                output.clear();
                first = false;
            }
            output.push((i, j));
        })?;
        if first {
            output.clear();
        }
        Ok(())
    }

    /// Visits unique cutoff-filtered pairs without allocating a pair vector.
    /// Requires unchanged positions since the last rebuild. Queries before the
    /// first rebuild or after count changes fail; arbitrary position edits cannot
    /// be detected because particle attributes remain publicly mutable.
    pub fn for_each_pair(
        &self,
        objects: &PhysObj,
        selection: ParticleSelection,
        mut visitor: impl FnMut(usize, usize),
    ) -> Result<(), ParticleNeighborListError> {
        let r = objects.core.get::<f64>(ATTR_R)?;
        self.validate_position_dim(r.dim())?;
        let (dim, n) = (r.dim(), r.num_vectors());
        if self.built_for != Some(n) {
            return Err(ParticleNeighborListError::Geometry {
                message:
                    "rebuild neighbor candidates for the current particle count before querying"
                        .to_string(),
            });
        }
        let alive = if !selection.includes_dead()
            && objects.core.contains(super::super::attrs::ATTR_ALIVE)
        {
            Some(objects.core.get::<u8>(super::super::attrs::ATTR_ALIVE)?)
        } else {
            None
        };
        let masks = BorrowedMasks::new([alive, None], n)?;
        let positions = r.borrow_values();
        let cutoff_squared = self.cutoff * self.cutoff;
        self.candidates.for_each_pair_candidate(|i, j| {
            if !masks.alive(i) || !masks.alive(j) {
                return;
            }
            let squared: f64 = positions[i * dim..][..dim]
                .iter()
                .zip(&positions[j * dim..][..dim])
                .map(|(a, b)| {
                    let difference = b - a;
                    difference * difference
                })
                .sum();
            if squared.is_finite() && squared > 0.0 && squared < cutoff_squared {
                visitor(i, j);
            }
        });
        Ok(())
    }

    fn validate_position_dim(&self, got_dim: usize) -> Result<(), ParticleNeighborListError> {
        let expected_dim = self.dim();
        if got_dim != expected_dim {
            return Err(ParticleStateError::InvalidAttrShape {
                label: ATTR_R,
                expected_dim,
                got_dim,
            }
            .into());
        }
        Ok(())
    }
}

/*!
Particle adapter for continuous boundary conditions.

Purpose:
This module applies the pure geometric boundaries from `space::continuous` to
canonical particle state. The geometry module knows how to wrap, clamp, or
reflect one coordinate vector. This adapter knows how particle state is stored:
positions live in `ATTR_R`, velocities live in `ATTR_V`, dead particles are
marked by `ATTR_ALIVE`, and rigid particles are marked by `ATTR_RIGID`.

Design:
The continuous boundary remains the single source of boundary mathematics.
Particle-specific behavior is limited to storage traversal and masks:
- dead particles are skipped;
- rigid particles are skipped;
- positions are updated in parallel;
- velocity components are flipped only when the continuous boundary reports an
  odd number of reflecting wall crossings.
*/

use core::fmt;

use crate::threading::try_for_each_pair_chunk_mut;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{ATTR_R, ATTR_V, ParticleSelection};
use crate::models::particles::state::{
    BorrowedMasks, ParticleStateError, mask_labels, validate_vector_attr_f64,
};
use crate::space::continuous::boundary::{
    BoundaryError as ContinuousBoundaryError, ContinuousBoundary,
};

#[derive(Debug, Clone, PartialEq)]
pub enum ParticleBoundaryError {
    Boundary(ContinuousBoundaryError),
    State(ParticleStateError),
}

impl From<AttrsError> for ParticleBoundaryError {
    #[inline]
    fn from(value: AttrsError) -> Self {
        Self::State(value.into())
    }
}

impl From<ContinuousBoundaryError> for ParticleBoundaryError {
    #[inline]
    fn from(value: ContinuousBoundaryError) -> Self {
        Self::Boundary(value)
    }
}

impl From<ParticleStateError> for ParticleBoundaryError {
    fn from(value: ParticleStateError) -> Self {
        Self::State(value)
    }
}

impl fmt::Display for ParticleBoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boundary(error) => write!(f, "particle boundary geometry error: {error}"),
            Self::State(error) => write!(f, "invalid particle state: {error}"),
        }
    }
}

impl std::error::Error for ParticleBoundaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Boundary(error) => Some(error),
            Self::State(error) => Some(error),
        }
    }
}

pub trait ParticleBoundary: ContinuousBoundary + Send + Sync {
    /// Apply this continuous boundary to canonical particle positions and
    /// velocities stored in `PhysObj`. Built-in boundaries borrow dense buffers.
    /// Fallible custom boundaries stage both columns and commit only on success;
    /// callback side effects outside particle state cannot be rolled back.
    fn apply_to_particles(&self, objects: &mut PhysObj) -> Result<(), ParticleBoundaryError>;
}

impl<T> ParticleBoundary for T
where
    T: ContinuousBoundary + Send + Sync,
{
    fn apply_to_particles(&self, objects: &mut PhysObj) -> Result<(), ParticleBoundaryError> {
        let r = objects.core.get::<f64>(ATTR_R)?;
        let (dim, n) = (r.dim(), r.num_vectors());
        validate_vector_attr_f64(objects, ATTR_V, dim, n)?;
        if self.dim() != dim {
            return Err(ContinuousBoundaryError::InvalidVectorDimension {
                label: "bounds",
                expected: dim,
                got: self.dim(),
            }
            .into());
        }
        let flags = mask_labels(objects, ParticleSelection::AliveOnly);
        let ([r, v], flags) = objects
            .core
            .get_mixed_mut::<f64, u8, 2, 2>([ATTR_R, ATTR_V], flags)?;
        let masks = BorrowedMasks::new(flags, n)?;
        let apply = |positions: &mut [f64], velocities: &mut [f64]| {
            try_for_each_pair_chunk_mut(
                positions,
                velocities,
                dim,
                |start, positions, velocities| {
                    for (index, (r, v)) in positions
                        .chunks_exact_mut(dim)
                        .zip(velocities.chunks_exact_mut(dim))
                        .enumerate()
                    {
                        if !masks.should_skip(start / dim + index) {
                            self.apply_position_velocity(r, v)?;
                        }
                    }
                    Ok::<_, ContinuousBoundaryError>(())
                },
            )
        };
        if self.may_fail_after_validation() {
            // Arbitrary callbacks can fail after earlier rows were updated.
            let mut positions = r.logical_values();
            let mut velocities = v.logical_values();
            apply(&mut positions, &mut velocities)?;
            r.replace_values(positions);
            v.replace_values(velocities);
        } else {
            r.edit_values(|positions| v.edit_values(|velocities| apply(positions, velocities)))?;
        }
        Ok(())
    }
}

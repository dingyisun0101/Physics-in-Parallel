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

use rayon::prelude::*;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{ATTR_R, ATTR_V, ParticleSelection};
use crate::models::particles::state::{
    ParticleMasks, ParticleStateError, gather_masks, validate_vector_attr_f64,
};
use crate::space::continuous::boundary::{
    BoundaryError as ContinuousBoundaryError, ContinuousBoundary,
};

#[derive(Debug, Clone, PartialEq)]
pub enum ParticleBoundaryError {
    Boundary(ContinuousBoundaryError),
    Attrs(AttrsError),
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

impl From<AttrsError> for ParticleBoundaryError {
    #[inline]
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
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

pub trait ParticleBoundary: ContinuousBoundary {
    /// Apply this continuous boundary to canonical particle positions and
    /// velocities stored in `PhysObj`.
    fn apply_to_particles(&self, objects: &mut PhysObj) -> Result<(), ParticleBoundaryError>;
}

impl<T> ParticleBoundary for T
where
    T: ContinuousBoundary,
{
    fn apply_to_particles(&self, objects: &mut PhysObj) -> Result<(), ParticleBoundaryError> {
        let (dim, n, masks) = shape_alive_rigid(objects)?;
        if self.dim() != dim {
            return Err(ContinuousBoundaryError::InvalidVectorDimension {
                label: "bounds",
                expected: dim,
                got: self.dim(),
            }
            .into());
        }

        let mut flip_mask: Vec<u8> = vec![0; n * dim];

        {
            let r = objects.core.get_mut::<f64>(ATTR_R)?;
            r.as_tensor_mut()
                .data
                .par_chunks_mut(dim)
                .zip(flip_mask.par_chunks_mut(dim))
                .enumerate()
                .try_for_each(|(i, (r_row, mask_row))| {
                    if masks.should_skip(i) {
                        return Ok(());
                    }
                    self.apply_position_with_velocity_flip_mask(r_row, mask_row)
                })?;
        }

        {
            let v = objects.core.get_mut::<f64>(ATTR_V)?;
            v.as_tensor_mut()
                .data
                .par_chunks_mut(dim)
                .zip(flip_mask.par_chunks(dim))
                .enumerate()
                .for_each(|(i, (v_row, mask_row))| {
                    if masks.should_skip(i) {
                        return;
                    }

                    for d in 0..dim {
                        if mask_row[d] == 1 {
                            v_row[d] = -v_row[d];
                        }
                    }
                });
        }

        Ok(())
    }
}

#[inline]
fn shape_alive_rigid(
    objects: &PhysObj,
) -> Result<(usize, usize, ParticleMasks), ParticleBoundaryError> {
    let (dim, n) = {
        let r = objects.core.get::<f64>(ATTR_R)?;
        (r.dim(), r.num_vectors())
    };

    validate_vector_attr_f64(objects, ATTR_V, dim, n)?;
    let masks = gather_masks(objects, n, ParticleSelection::AliveOnly)?;
    Ok((dim, n, masks))
}

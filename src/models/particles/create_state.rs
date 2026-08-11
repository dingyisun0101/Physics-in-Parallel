/*!
Canonical massive-particle state construction and randomization.

Purpose:
This module builds `PhysObj` instances with the particle attributes expected by
the rest of the particle model code. Generic continuous-vector randomization
delegates to `space::continuous::sampling`; the Maxwell-Boltzmann velocity path
remains here because it depends on particle inverse-mass attributes.

Canonical template attributes:
- `r`: position vector, initialized to zero.
- `v`: velocity vector, initialized to zero.
- `a`: acceleration vector, initialized to zero.
- `m`: mass scalar, initialized to one.
- `m_inv`: inverse-mass scalar, initialized to one.
- `alive`: alive-mask scalar, initialized to one.
- `rigid`: rigid-mask scalar, initialized to zero.
*/

use crate::math::tensor::rank_2::vector_list::VectorList;
use crate::math::tensor::{RandType, TensorRandFiller};
use crate::space::continuous::sampling::{
    VectorSamplingError, VectorSamplingMethod, sample_vectors,
};
use rayon::prelude::*;

use crate::engines::soa::phys_obj::{AttrsCore, AttrsError, AttrsMeta, PhysObj};
pub use crate::models::particles::attrs::{
    ALIVE_TRUE, ATTR_A, ATTR_ALIVE, ATTR_M, ATTR_M_INV, ATTR_R, ATTR_RIGID, ATTR_V, RIGID_FALSE,
};
use crate::models::particles::state::{ParticleStateError, gather_inverse_mass};

/// Errors returned by massive-particle construction and randomization helpers.
#[derive(Debug, Clone, PartialEq)]
pub enum MassiveParticlesError {
    /// Lower-level attribute/core error.
    Attrs(AttrsError),
    /// Requested particle vector dimension is zero.
    InvalidDimension {
        /// Invalid dimension value.
        dim: usize,
    },
    /// Requested particle count is zero.
    InvalidParticleCount {
        /// Invalid particle count.
        n: usize,
    },
    /// Temperature-like Maxwell-Boltzmann parameter is not finite or negative.
    InvalidTau {
        /// Invalid tau value.
        tau: f64,
    },
    /// Inverse mass is not finite or is negative.
    InvalidMassInv {
        /// Particle row index.
        index: usize,
        /// Invalid inverse-mass value.
        value: f64,
    },
    /// Inverse-mass attribute is not scalar-valued.
    InvalidMassInvShape {
        /// Expected vector dimension.
        expected_dim: usize,
        /// Actual vector dimension.
        got_dim: usize,
    },
    /// Attribute row count does not match the velocity row count.
    InconsistentParticleCount {
        /// Expected number of rows.
        expected: usize,
        /// Actual number of rows.
        got: usize,
    },
    /// Distribution parameters are invalid.
    Distribution {
        /// Human-readable validation message.
        msg: String,
    },
}

impl From<AttrsError> for MassiveParticlesError {
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
}

impl From<VectorSamplingError> for MassiveParticlesError {
    fn from(value: VectorSamplingError) -> Self {
        Self::Distribution {
            msg: format!("{value:?}"),
        }
    }
}

impl From<ParticleStateError> for MassiveParticlesError {
    fn from(value: ParticleStateError) -> Self {
        match value {
            ParticleStateError::Attrs(err) => Self::Attrs(err),
            ParticleStateError::InvalidAttrShape {
                expected_dim,
                got_dim,
                ..
            } => Self::InvalidMassInvShape {
                expected_dim,
                got_dim,
            },
            ParticleStateError::InconsistentParticleCount { expected, got, .. } => {
                Self::InconsistentParticleCount { expected, got }
            }
        }
    }
}

/// Constructs a canonical massive-particle `PhysObj`.
///
/// `dim` is the vector dimension used by position, velocity, and acceleration.
/// `num_particles` is the shared row count for all attributes.
pub fn create_template(dim: usize, num_particles: usize) -> Result<PhysObj, MassiveParticlesError> {
    if dim == 0 {
        return Err(MassiveParticlesError::InvalidDimension { dim });
    }
    if num_particles == 0 {
        return Err(MassiveParticlesError::InvalidParticleCount { n: num_particles });
    }

    let mut core = AttrsCore::empty();

    core.allocate::<f64>(ATTR_R, dim, num_particles)?;
    core.allocate::<f64>(ATTR_V, dim, num_particles)?;
    core.allocate::<f64>(ATTR_A, dim, num_particles)?;

    // Scalar-valued fields represented as dim=1 vector-lists.
    let mut m = VectorList::<f64>::empty(1, num_particles);
    m.fill(1.0);
    core.insert(ATTR_M, m)?;

    let mut m_inv = VectorList::<f64>::empty(1, num_particles);
    m_inv.fill(1.0);
    core.insert(ATTR_M_INV, m_inv)?;

    let mut alive = VectorList::<u8>::empty(1, num_particles);
    alive.fill(ALIVE_TRUE);
    core.insert(ATTR_ALIVE, alive)?;

    let mut rigid = VectorList::<u8>::empty(1, num_particles);
    rigid.fill(RIGID_FALSE);
    core.insert(ATTR_RIGID, rigid)?;

    let meta = AttrsMeta::new(0, "particles", "canonical massive-particle state");

    Ok(PhysObj::new(meta, core))
}

/// Randomizes particle positions in `ATTR_R`.
pub fn randomize_r(
    phys_obj: &mut PhysObj,
    method: VectorSamplingMethod<'_>,
) -> Result<(), MassiveParticlesError> {
    let r = phys_obj.core.get_mut::<f64>(ATTR_R)?;
    sample_vectors(r, method)?;
    Ok(())
}

/// Velocity sampling strategy for canonical particle velocities.
#[derive(Debug, Clone, PartialEq)]
pub enum VelocitySamplingMethod<'a> {
    /// Uniform velocity components in `[low, high)`.
    Uniform { low: f64, high: f64 },
    /// Maxwell-Boltzmann-like Gaussian components with variance `tau * m_inv`.
    MaxwellBoltzmann { tau: f64 },
    /// Independent Gaussian components with per-axis mean and standard deviation.
    GaussianPerAxis {
        /// Per-axis mean.
        mean: &'a [f64],
        /// Per-axis standard deviation.
        std: &'a [f64],
    },
}

/// Randomizes particle velocities in `ATTR_V`.
pub fn randomize_v(
    phys_obj: &mut PhysObj,
    method: VelocitySamplingMethod<'_>,
) -> Result<(), MassiveParticlesError> {
    match method {
        VelocitySamplingMethod::Uniform { low, high } => {
            let v = phys_obj.core.get_mut::<f64>(ATTR_V)?;
            sample_vectors(v, VectorSamplingMethod::Uniform { low, high })?;
            Ok(())
        }
        VelocitySamplingMethod::GaussianPerAxis { mean, std } => {
            let v = phys_obj.core.get_mut::<f64>(ATTR_V)?;
            sample_vectors(v, VectorSamplingMethod::GaussianPerAxis { mean, std })?;
            Ok(())
        }
        VelocitySamplingMethod::MaxwellBoltzmann { tau } => {
            if !tau.is_finite() || tau < 0.0 {
                return Err(MassiveParticlesError::InvalidTau { tau });
            }

            let (dim, n) = {
                let v = phys_obj.core.get::<f64>(ATTR_V)?;
                (v.dim(), v.num_vectors())
            };

            let m_inv_values = gather_inverse_mass(phys_obj, n)?;
            for (i, &m_inv_i) in m_inv_values.iter().enumerate() {
                if !m_inv_i.is_finite() || m_inv_i < 0.0 {
                    return Err(MassiveParticlesError::InvalidMassInv {
                        index: i,
                        value: m_inv_i,
                    });
                }
            }

            let v = phys_obj.core.get_mut::<f64>(ATTR_V)?;
            if tau == 0.0 {
                v.as_tensor_mut().data.par_iter_mut().for_each(|x| *x = 0.0);
                return Ok(());
            }

            let mut filler = TensorRandFiller::new(
                RandType::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
                None,
                None,
                None,
            );
            filler.refresh(v.as_tensor_mut());

            v.as_tensor_mut()
                .data
                .par_chunks_mut(dim)
                .zip(m_inv_values.par_iter())
                .for_each(|(row, &m_inv_i)| {
                    let sigma = (tau * m_inv_i).sqrt();
                    if sigma == 0.0 {
                        for x in row.iter_mut() {
                            *x = 0.0;
                        }
                        return;
                    }

                    for x in row.iter_mut().take(dim) {
                        *x *= sigma;
                    }
                });

            Ok(())
        }
    }
}

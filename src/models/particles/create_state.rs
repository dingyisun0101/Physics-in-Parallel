/*!
Canonical massive-particle state construction and randomization.

Purpose:
This module builds `PhysObj` instances with the particle attributes expected by
the rest of the particle model code. It also provides common position and
velocity randomizers that operate on those canonical attributes.

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
use rayon::prelude::*;

use crate::engines::soa::phys_obj::{AttrsCore, AttrsError, AttrsMeta, PhysObj};
pub use crate::models::particles::attrs::{
    ALIVE_TRUE, ATTR_A, ATTR_ALIVE, ATTR_M, ATTR_M_INV, ATTR_R, ATTR_RIGID, ATTR_V,
};

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

    let rigid = VectorList::<f64>::empty(1, num_particles);
    core.insert(ATTR_RIGID, rigid)?;

    let meta = AttrsMeta::new(0, "particles", "canonical massive-particle state");

    Ok(PhysObj::new(meta, core))
}

/// Position randomization strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum RandPosMethod<'a> {
    /// Uniform random placement centered at zero.
    ///
    /// Each coordinate is sampled in `[-box_size[k] / 2, box_size[k] / 2]`.
    Uniform {
        /// Full box width on each axis.
        box_size: &'a [f64],
    },
    /// Regular lattice coordinate plus independent Gaussian jitter per axis.
    JitteredLattice {
        /// Lattice spacing on each axis.
        spacings: &'a [f64],
        /// Gaussian standard deviation on each axis.
        sigmas: &'a [f64],
    },
}

/// Randomizes particle positions in `ATTR_R`.
pub fn randomize_r(
    phys_obj: &mut PhysObj,
    method: RandPosMethod<'_>,
) -> Result<(), MassiveParticlesError> {
    let (dim, n) = {
        let r = phys_obj.core.get::<f64>(ATTR_R)?;
        (r.dim(), r.num_vectors())
    };

    if dim == 0 || n == 0 {
        return Ok(());
    }

    let r = phys_obj.core.get_mut::<f64>(ATTR_R)?;

    match method {
        RandPosMethod::Uniform { box_size } => {
            if box_size.len() != dim {
                return Err(MassiveParticlesError::Distribution {
                    msg: format!(
                        "Uniform.box_size length mismatch: expected {}, got {}",
                        dim,
                        box_size.len()
                    ),
                });
            }
            for (k, &size) in box_size.iter().enumerate() {
                if !size.is_finite() || size < 0.0 {
                    return Err(MassiveParticlesError::Distribution {
                        msg: format!(
                            "Uniform.box_size[{}] must be finite and non-negative, got {}",
                            k, size
                        ),
                    });
                }
            }

            let mut filler = TensorRandFiller::new(
                RandType::Uniform {
                    low: 0.0,
                    high: 1.0,
                },
                None,
            );
            filler.refresh(r.as_tensor_mut());

            r.as_tensor_mut().data.par_chunks_mut(dim).for_each(|row| {
                for k in 0..dim {
                    let half_span = 0.5 * box_size[k];
                    row[k] = (2.0 * row[k] - 1.0) * half_span;
                }
            });
        }
        RandPosMethod::JitteredLattice { spacings, sigmas } => {
            if spacings.len() != dim {
                return Err(MassiveParticlesError::Distribution {
                    msg: format!(
                        "JitteredGrid.spacings length mismatch: expected {}, got {}",
                        dim,
                        spacings.len()
                    ),
                });
            }
            if sigmas.len() != dim {
                return Err(MassiveParticlesError::Distribution {
                    msg: format!(
                        "JitteredGrid.sigmas length mismatch: expected {}, got {}",
                        dim,
                        sigmas.len()
                    ),
                });
            }
            for (k, &spacing) in spacings.iter().enumerate() {
                if !spacing.is_finite() || spacing < 0.0 {
                    return Err(MassiveParticlesError::Distribution {
                        msg: format!(
                            "JitteredGrid.spacings[{}] must be finite and non-negative, got {}",
                            k, spacing
                        ),
                    });
                }
            }
            for (k, &sigma) in sigmas.iter().enumerate() {
                if !sigma.is_finite() || sigma < 0.0 {
                    return Err(MassiveParticlesError::Distribution {
                        msg: format!(
                            "JitteredGrid.sigmas[{}] must be finite and non-negative, got {}",
                            k, sigma
                        ),
                    });
                }
            }

            let mut filler = TensorRandFiller::new(
                RandType::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
                None,
            );
            filler.refresh(r.as_tensor_mut());

            let side = ((n as f64).powf(1.0 / dim as f64).ceil() as usize).max(1);
            r.as_tensor_mut()
                .data
                .par_chunks_mut(dim)
                .enumerate()
                .for_each(|(particle_idx, row)| {
                    let mut lattice_idx = particle_idx;
                    for k in 0..dim {
                        let grid_coord = lattice_idx % side;
                        lattice_idx /= side;
                        let base = grid_coord as f64 * spacings[k];
                        row[k] = base + row[k] * sigmas[k];
                    }
                });
        }
    }

    Ok(())
}

/// Velocity randomization strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum RandVelMethod<'a> {
    /// Uniform velocity components in `[low, high)`.
    Uniform { low: f64, high: f64 },
    /// Maxwell-Boltzmann-like Gaussian components with variance `tau * m_inv`.
    MaxwellBoltzmann { tau: f64 },
    /// Independent Gaussian components with per-axis mean and standard deviation.
    DriftGaussian { avg: &'a [f64], sigma: &'a [f64] },
}

/// Randomizes particle velocities in `ATTR_V`.
pub fn randomize_v(
    phys_obj: &mut PhysObj,
    method: RandVelMethod<'_>,
) -> Result<(), MassiveParticlesError> {
    match method {
        RandVelMethod::Uniform { low, high } => {
            if !low.is_finite() || !high.is_finite() || low >= high {
                return Err(MassiveParticlesError::Distribution {
                    msg: format!(
                        "Uniform velocity bounds must be finite with low < high; got low={}, high={}",
                        low, high
                    ),
                });
            }

            let v = phys_obj.core.get_mut::<f64>(ATTR_V)?;
            let mut filler = TensorRandFiller::new(RandType::Uniform { low, high }, None);
            filler.refresh(v.as_tensor_mut());
            Ok(())
        }
        RandVelMethod::DriftGaussian { avg, sigma } => {
            let dim = {
                let v = phys_obj.core.get::<f64>(ATTR_V)?;
                v.dim()
            };

            if avg.len() != dim {
                return Err(MassiveParticlesError::Distribution {
                    msg: format!(
                        "DriftGaussian.avg length mismatch: expected {}, got {}",
                        dim,
                        avg.len()
                    ),
                });
            }
            if sigma.len() != dim {
                return Err(MassiveParticlesError::Distribution {
                    msg: format!(
                        "DriftGaussian.sigma length mismatch: expected {}, got {}",
                        dim,
                        sigma.len()
                    ),
                });
            }
            for (k, &a) in avg.iter().enumerate() {
                if !a.is_finite() {
                    return Err(MassiveParticlesError::Distribution {
                        msg: format!("DriftGaussian.avg[{}] must be finite, got {}", k, a),
                    });
                }
            }
            for (k, &s) in sigma.iter().enumerate() {
                if !s.is_finite() || s < 0.0 {
                    return Err(MassiveParticlesError::Distribution {
                        msg: format!(
                            "DriftGaussian.sigma[{}] must be finite and non-negative, got {}",
                            k, s
                        ),
                    });
                }
            }

            let v = phys_obj.core.get_mut::<f64>(ATTR_V)?;
            let mut filler = TensorRandFiller::new(
                RandType::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
                None,
            );
            filler.refresh(v.as_tensor_mut());

            v.as_tensor_mut().data.par_chunks_mut(dim).for_each(|row| {
                for k in 0..dim {
                    row[k] = avg[k] + row[k] * sigma[k];
                }
            });

            Ok(())
        }
        RandVelMethod::MaxwellBoltzmann { tau } => {
            if !tau.is_finite() || tau < 0.0 {
                return Err(MassiveParticlesError::InvalidTau { tau });
            }

            let (dim, n) = {
                let v = phys_obj.core.get::<f64>(ATTR_V)?;
                (v.dim(), v.num_vectors())
            };

            let m_inv_values: Vec<f64> = {
                let m_inv = phys_obj.core.get::<f64>(ATTR_M_INV)?;

                if m_inv.dim() != 1 {
                    return Err(MassiveParticlesError::InvalidMassInvShape {
                        expected_dim: 1,
                        got_dim: m_inv.dim(),
                    });
                }
                if m_inv.num_vectors() != n {
                    return Err(MassiveParticlesError::InconsistentParticleCount {
                        expected: n,
                        got: m_inv.num_vectors(),
                    });
                }

                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let v = m_inv.get(i as isize, 0);
                    if !v.is_finite() || v < 0.0 {
                        return Err(MassiveParticlesError::InvalidMassInv { index: i, value: v });
                    }
                    out.push(v);
                }
                out
            };

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

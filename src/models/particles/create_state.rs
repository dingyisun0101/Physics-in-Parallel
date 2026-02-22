/*!
Massive-particle convenience constructors on top of `PhysObj`.
*/

use crate::math::tensor::rank_2::vector_list::VectorList;
use crate::math::tensor::core::dense_rand::{RandType, TensorRandFiller};
use rayon::prelude::*;

use crate::engines::soa::phys_obj::{AttrsCore, AttrsError, AttrsMeta, PhysObj};
pub use crate::models::particles::attrs::{
    ATTR_A,
    ATTR_M,
    ATTR_M_INV,
    ATTR_R,
    ATTR_V,
};

#[derive(Debug, Clone, PartialEq)]
pub enum MassiveParticlesError {
    Attrs(AttrsError),
    InvalidTau { tau: f64 },
    InvalidMassInv { index: usize, value: f64 },
    InvalidMassInvShape { expected_dim: usize, got_dim: usize },
    InconsistentParticleCount { expected: usize, got: usize },
    Distribution { msg: String },
}

impl From<AttrsError> for MassiveParticlesError {
    #[inline]
    /// - Purpose: Converts an `AttrsError` into a `MassiveParticlesError`.
    /// - Parameters:
    ///   - `value` (`AttrsError`): Attribute-layer error to wrap.
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
}







// =============================================================================
// -------------------------- Empty Instance Creator ---------------------------
// =============================================================================
#[inline]
/// - Purpose: Constructs a `PhysObj` configured with canonical massive-particle core fields.
/// - Parameters:
///   - `dim` (`usize`): Vector dimension used for `r`, `v`, and `a`.
///   - `num_particles` (`usize`): Number of particles (row count for all attributes).
pub fn create_template(dim: usize, num_particles: usize) -> Result<PhysObj, AttrsError> {
    let mut core = AttrsCore::empty();

    // Vector-valued fields: initialized to 0 by VectorList::empty.
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

    let meta = AttrsMeta {
        id: 0,
        label: "particles".to_string(),
        comment: String::new(),
    };

    Ok(PhysObj { meta, core })
}







// =============================================================================
// --------------------- Particle States Randomizers ---------------------------
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum RandPosMethod<'a> {
    Uniform { box_size: &'a [f64]}, // truly random asignment for each particle
    JitteredGrid {spacings: &'a [f64], sigmas: &'a [f64]}, // Grid site + random jitter
}




#[inline]
/// - Purpose: Randomizes particle positions (`r`) using the requested placement strategy.
/// - Parameters:
///   - `phys_obj` (`&mut PhysObj`): Particle container whose `r` field will be overwritten.
///   - `method` (`RandPosMethod<'_>`): Position randomization strategy and its method-specific parameters.
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
                RandType::Uniform { low: 0.0, high: 1.0 },
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
        RandPosMethod::JitteredGrid { spacings, sigmas } => {
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
                RandType::Normal { mean: 0.0, std: 1.0 },
                None,
            );
            filler.refresh(r.as_tensor_mut());

            let side = ((n as f64).powf(1.0 / dim as f64).ceil() as usize).max(1);
            r.as_tensor_mut().data.par_chunks_mut(dim)
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


pub enum RandVelMethod<'a> {
    Uniform {low: f64, high: f64},
    MaxwellBoltzmann {tau: f64},
    DriftGaussian { avg: &'a [f64], sigma: &'a [f64] },
}

#[inline]
/// - Purpose: Randomizes particle velocities using the requested velocity randomization strategy.
/// - Parameters:
///   - `phys_obj` (`&mut PhysObj`): Particle container whose `v` field will be overwritten.
///   - `method` (`RandVelMethod<'_>`): Velocity randomization strategy and its method-specific parameters.
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
            let mut filler = TensorRandFiller::new(
                RandType::Uniform { low, high },
                None,
            );
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
                RandType::Normal { mean: 0.0, std: 1.0 },
                None,
            );
            filler.refresh(v.as_tensor_mut());

            v.as_tensor_mut()
                .data
                .par_chunks_mut(dim)
                .for_each(|row| {
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
                v.as_tensor_mut()
                    .data
                    .par_iter_mut()
                    .for_each(|x| *x = 0.0);
                return Ok(());
            }

            let mut filler = TensorRandFiller::new(
                RandType::Normal { mean: 0.0, std: 1.0 },
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

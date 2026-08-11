/*!
Continuous-space vector sampling.

Purpose:
This module fills a list of real-valued vectors with common initialization
patterns. It is independent of particles and attribute labels: callers provide a
`VectorList<f64>` and choose a sampling method. Particle state construction uses
the same sampler for positions and for velocity distributions that do not
depend on particle mass.

Data shape:
The target storage is a `VectorList<f64>` with shape `[n_vectors, dim]`. Each
row is one sampled continuous vector.
*/

use rayon::prelude::*;

use crate::math::tensor::rank_2::vector_list::VectorList;
use crate::math::tensor::{RandType, TensorRandError, TensorRandFiller};
use crate::rng::RngConfig;

#[derive(Debug, Clone, PartialEq)]
pub enum VectorSamplingMethod<'a> {
    /// Uniform components in `[low, high)`.
    Uniform { low: f64, high: f64 },
    /// Uniform random placement centered at zero.
    ///
    /// Each coordinate is sampled in `[-box_size[k] / 2, box_size[k] / 2]`.
    UniformCentered {
        /// Full box width on each axis.
        box_size: &'a [f64],
    },
    /// Independent Gaussian components with per-axis mean and standard deviation.
    GaussianPerAxis {
        /// Per-axis mean.
        mean: &'a [f64],
        /// Per-axis standard deviation.
        std: &'a [f64],
    },
    /// Regular lattice coordinate plus independent Gaussian jitter per axis.
    JitteredLattice {
        /// Lattice spacing on each axis.
        spacings: &'a [f64],
        /// Gaussian standard deviation on each axis.
        sigmas: &'a [f64],
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum VectorSamplingError {
    InvalidUniformBounds {
        low: f64,
        high: f64,
    },
    InvalidParameterLength {
        parameter: &'static str,
        expected: usize,
        got: usize,
    },
    InvalidParameterValue {
        parameter: &'static str,
        index: usize,
        value: f64,
        rule: &'static str,
    },
    Rng(TensorRandError),
}

pub fn sample_vectors(
    vectors: &mut VectorList<f64>,
    method: VectorSamplingMethod<'_>,
    rng: RngConfig,
) -> Result<RngConfig, VectorSamplingError> {
    let dim = vectors.dim();
    let n = vectors.num_vectors();

    let resolved = match method {
        VectorSamplingMethod::Uniform { low, high } => {
            if !low.is_finite() || !high.is_finite() || low >= high {
                return Err(VectorSamplingError::InvalidUniformBounds { low, high });
            }

            let mut filler = TensorRandFiller::try_new(RandType::Uniform { low, high }, rng)
                .map_err(VectorSamplingError::Rng)?;
            if dim > 0 && n > 0 {
                filler
                    .try_refresh(vectors.as_tensor_mut())
                    .map_err(VectorSamplingError::Rng)?;
            }
            filler.rng_config()
        }
        VectorSamplingMethod::UniformCentered { box_size } => {
            validate_len("box_size", box_size.len(), dim)?;
            validate_finite_nonnegative("box_size", box_size)?;

            let mut filler = TensorRandFiller::try_new(
                RandType::Uniform {
                    low: 0.0,
                    high: 1.0,
                },
                rng,
            )
            .map_err(VectorSamplingError::Rng)?;
            if dim > 0 && n > 0 {
                filler
                    .try_refresh(vectors.as_tensor_mut())
                    .map_err(VectorSamplingError::Rng)?;
            }

            if dim > 0 && n > 0 {
                vectors
                    .as_tensor_mut()
                    .data
                    .par_chunks_mut(dim)
                    .for_each(|row| {
                        for k in 0..dim {
                            let half_span = 0.5 * box_size[k];
                            row[k] = (2.0 * row[k] - 1.0) * half_span;
                        }
                    });
            }
            filler.rng_config()
        }
        VectorSamplingMethod::GaussianPerAxis { mean, std } => {
            validate_len("mean", mean.len(), dim)?;
            validate_len("std", std.len(), dim)?;
            validate_finite("mean", mean)?;
            validate_finite_nonnegative("std", std)?;

            let mut filler = TensorRandFiller::try_new(
                RandType::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
                rng,
            )
            .map_err(VectorSamplingError::Rng)?;
            if dim > 0 && n > 0 {
                filler
                    .try_refresh(vectors.as_tensor_mut())
                    .map_err(VectorSamplingError::Rng)?;
            }

            if dim > 0 && n > 0 {
                vectors
                    .as_tensor_mut()
                    .data
                    .par_chunks_mut(dim)
                    .for_each(|row| {
                        for k in 0..dim {
                            row[k] = mean[k] + row[k] * std[k];
                        }
                    });
            }
            filler.rng_config()
        }
        VectorSamplingMethod::JitteredLattice { spacings, sigmas } => {
            validate_len("spacings", spacings.len(), dim)?;
            validate_len("sigmas", sigmas.len(), dim)?;
            validate_finite_nonnegative("spacings", spacings)?;
            validate_finite_nonnegative("sigmas", sigmas)?;

            let mut filler = TensorRandFiller::try_new(
                RandType::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
                rng,
            )
            .map_err(VectorSamplingError::Rng)?;
            if dim > 0 && n > 0 {
                filler
                    .try_refresh(vectors.as_tensor_mut())
                    .map_err(VectorSamplingError::Rng)?;
            }

            if dim > 0 && n > 0 {
                let side = ((n as f64).powf(1.0 / dim as f64).ceil() as usize).max(1);
                vectors
                    .as_tensor_mut()
                    .data
                    .par_chunks_mut(dim)
                    .enumerate()
                    .for_each(|(vector_idx, row)| {
                        let mut lattice_idx = vector_idx;
                        for k in 0..dim {
                            let grid_coord = lattice_idx % side;
                            lattice_idx /= side;
                            let base = grid_coord as f64 * spacings[k];
                            row[k] = base + row[k] * sigmas[k];
                        }
                    });
            }
            filler.rng_config()
        }
    };

    Ok(resolved)
}

fn validate_len(
    parameter: &'static str,
    got: usize,
    expected: usize,
) -> Result<(), VectorSamplingError> {
    if got != expected {
        return Err(VectorSamplingError::InvalidParameterLength {
            parameter,
            expected,
            got,
        });
    }
    Ok(())
}

fn validate_finite(parameter: &'static str, values: &[f64]) -> Result<(), VectorSamplingError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(VectorSamplingError::InvalidParameterValue {
                parameter,
                index,
                value,
                rule: "finite",
            });
        }
    }
    Ok(())
}

fn validate_finite_nonnegative(
    parameter: &'static str,
    values: &[f64],
) -> Result<(), VectorSamplingError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(VectorSamplingError::InvalidParameterValue {
                parameter,
                index,
                value,
                rule: "finite and non-negative",
            });
        }
    }
    Ok(())
}

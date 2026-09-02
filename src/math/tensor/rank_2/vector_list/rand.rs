/*!
Random vector-list generators.

Purpose:
    This module provides reusable random fillers for `VectorList` values while
    keeping random-number generation inside the rank-N tensor random
    infrastructure. The generators own their `VectorList` output and call
    `TensorRandFiller` to refresh dense buffers in-place.

Design:
    - `HaarVectors`:
        Uses exact random signs in rank one, angle parameterizations in ranks
        two and three, and normalized standard-normal rows in higher ranks.
        Each retrieved row is one Haar-distributed unit vector.
    - `NNVectors`:
        Fills a rank-N dense code tensor with exact integer direction codes
        using `TensorRandFiller`, then decodes those codes in parallel. The
        stored rows are one-hot nearest-neighbor directions.
*/
use crate::threading::parallel_chunk_len;
use rayon::prelude::*;

use crate::math::tensor::rank_n::dense::Tensor;
use crate::math::tensor::rank_n::tensor_trait::TensorTrait;
use crate::math::{RandType, TensorRandError, TensorRandFiller};
use crate::rng::ResolvedRng;

use super::VectorList;

// ============================================================================
// -------------------------- Haar-random unit vectors ------------------------
// ============================================================================
#[derive(Debug, Clone)]
pub struct HaarVectors {
    pub vl: VectorList<f64>,
    pub dim: usize,
    filler: TensorRandFiller,
    workspace: HaarWorkspace,
}

#[derive(Debug, Clone)]
enum HaarWorkspace {
    Sign { codes: Tensor<usize> },
    Plane { units: Vec<f64> },
    Sphere { units: Vec<f64> },
    Gaussian,
}

impl HaarVectors {
    /// Constructs schedule-independent Haar vectors for explicit scientific steps.
    pub fn try_new_indexed(
        dim: usize,
        n: usize,
        rng: ResolvedRng,
    ) -> Result<Self, TensorRandError> {
        assert!(dim > 0, "HaarVectors::new: dim must be > 0");
        assert!(n > 0, "HaarVectors::new: n must be > 0");
        let (kind, workspace) = match dim {
            1 => (
                RandType::UniformInt { low: 0, high: 1 },
                HaarWorkspace::Sign {
                    codes: Tensor::empty(&[n]),
                },
            ),
            2 => (
                RandType::Uniform {
                    low: 0.0,
                    high: 1.0,
                },
                HaarWorkspace::Plane {
                    units: vec![0.0; n],
                },
            ),
            3 => (
                RandType::Uniform {
                    low: 0.0,
                    high: 1.0,
                },
                HaarWorkspace::Sphere {
                    units: vec![0.0; n * 2],
                },
            ),
            _ => (
                RandType::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
                HaarWorkspace::Gaussian,
            ),
        };
        let filler = TensorRandFiller::new(kind, rng)?;
        Ok(Self {
            vl: VectorList::empty(dim, n),
            dim,
            filler,
            workspace,
        })
    }

    /// Refreshes schedule-independent Haar vectors for an explicit step and domain.
    pub fn try_refresh_at(&mut self, step: u64, domain: u64) -> Result<(), TensorRandError> {
        match &mut self.workspace {
            HaarWorkspace::Sign { codes } => {
                self.filler
                    .try_fill_indices_at(codes.data_mut(), 2, step, domain)?;
                write_signs(codes.data(), self.vl.as_tensor_mut().data_mut());
            }
            HaarWorkspace::Plane { units } => {
                self.filler.try_fill_slice_at(units, step, domain)?;
                write_plane(units, self.vl.as_tensor_mut().data_mut());
            }
            HaarWorkspace::Sphere { units } => {
                self.filler
                    .try_fill_slice_at_layout(units, 2, step, domain)?;
                write_sphere(units, self.vl.as_tensor_mut().data_mut());
            }
            HaarWorkspace::Gaussian => {
                self.filler.try_fill_slice_at_layout(
                    self.vl.as_tensor_mut().data_mut(),
                    self.dim,
                    step,
                    domain,
                )?;
                self.vl.normalize();
            }
        }
        Ok(())
    }
}

fn write_signs(codes: &[usize], output: &mut [f64]) {
    let min_vectors_per_job = parallel_chunk_len(output.len()).unwrap_or(1);
    output
        .par_iter_mut()
        .zip(codes.par_iter())
        .with_min_len(min_vectors_per_job)
        .for_each(|(value, code)| *value = if *code == 0 { -1.0 } else { 1.0 });
}

fn write_plane(units: &[f64], output: &mut [f64]) {
    output
        .par_chunks_exact_mut(2)
        .zip(units.par_iter())
        .with_min_len(parallel_chunk_len(units.len()).unwrap_or(1))
        .for_each(|(row, unit)| {
            let (sin, cos) = (std::f64::consts::TAU * unit).sin_cos();
            row[0] = cos;
            row[1] = sin;
        });
}

fn write_sphere(units: &[f64], output: &mut [f64]) {
    let min_vectors_per_job = parallel_chunk_len(output.len() / 3).unwrap_or(1);
    output
        .par_chunks_exact_mut(3)
        .zip(units.par_chunks_exact(2))
        .with_min_len(min_vectors_per_job)
        .for_each(|(row, random)| {
            let z = 2.0 * random[0] - 1.0;
            let radius = (1.0 - z * z).sqrt();
            let (sin, cos) = (std::f64::consts::TAU * random[1]).sin_cos();
            row[0] = radius * cos;
            row[1] = radius * sin;
            row[2] = z;
        });
}

// ============================================================================
// -------------------- Nearest-Neighbor one-hot ±1 vectors -------------------
// ============================================================================

#[derive(Debug, Clone)]
pub struct NNVectors {
    pub vl: VectorList<isize>, // shape [n, dim], entries in {-1, 0, +1}
    pub dim: usize,
    code_buf: Tensor<usize>,       // shape [n], holds codes in [0, 2*dim)
    code_filler: TensorRandFiller, // RandType::UniformInt over code range
}

#[inline]
fn decode_nearest_neighbor_code(code: usize, row: &mut [isize]) {
    let axis = code / 2;
    let sign = if code.is_multiple_of(2) {
        1isize
    } else {
        -1isize
    };
    for (k, x) in row.iter_mut().enumerate() {
        *x = if k == axis { sign } else { 0 };
    }
}

fn decode_nearest_neighbor_codes(codes: &[usize], dim: usize, output: &mut [isize]) {
    output
        .par_chunks_exact_mut(dim)
        .zip(codes.par_iter())
        .with_min_len(parallel_chunk_len(codes.len()).unwrap_or(1))
        .for_each(|(row, code)| decode_nearest_neighbor_code(*code, row));
}

impl NNVectors {
    /// Constructs schedule-independent nearest-neighbor vectors.
    pub fn try_new_indexed(
        dim: usize,
        n: usize,
        rng: ResolvedRng,
    ) -> Result<Self, TensorRandError> {
        assert!(dim > 0, "NNVectors::try_new_indexed: dim must be > 0");
        assert!(n > 0, "NNVectors::try_new_indexed: n must be > 0");
        Ok(Self {
            vl: VectorList::empty(dim, n),
            dim,
            code_buf: Tensor::empty(&[n]),
            code_filler: TensorRandFiller::new(
                RandType::UniformInt {
                    low: 0,
                    high: (2 * dim) as i64 - 1,
                },
                rng,
            )?,
        })
    }

    /// Refreshes schedule-independent nearest-neighbor vectors.
    pub fn try_refresh_at(&mut self, step: u64, domain: u64) -> Result<(), TensorRandError> {
        self.code_filler.try_fill_indices_at(
            self.code_buf.data_mut(),
            self.dim * 2,
            step,
            domain,
        )?;
        decode_nearest_neighbor_codes(
            self.code_buf.data(),
            self.dim,
            self.vl.as_tensor_mut().data_mut(),
        );
        Ok(())
    }
}

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
use ndarray::Array2;
use rayon::prelude::*;
use serde_json::Value;

use crate::math::tensor::{
    RandType, TensorRandError, TensorRandFiller, TensorTrait, dense::Tensor,
};
use crate::rng::RngConfig;

use super::VectorList;

// ============================================================================
// ------------------------------- Common Trait -------------------------------
// ============================================================================

/// Minimal interface for random vector-list generators.
pub trait VectorListRand {
    type Elem;

    /// Allocate output storage and rank-N random-fill buffers.
    fn new(dim: usize, n: usize, rng: RngConfig) -> Result<Self, TensorRandError>
    where
        Self: Sized;

    /// Refill the internal `VectorList` in-place, keeping shape `[n, dim]`.
    fn refresh(&mut self);
}

// ============================================================================
// -------------------------- Haar-random unit vectors ------------------------
// ============================================================================
#[derive(Debug, Clone)]
pub struct HaarVectors {
    pub vl: VectorList<f64>,
    pub dim: usize,
    pub n: usize,
    filler: TensorRandFiller,
    mode: FillMode,
    workspace: HaarWorkspace,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum FillMode {
    Stateful,
    Indexed,
}

#[derive(Debug, Clone)]
enum HaarWorkspace {
    Sign { codes: Tensor<usize> },
    Plane { units: Vec<f64> },
    Sphere { units: Vec<f64> },
    Gaussian,
}

impl VectorListRand for HaarVectors {
    type Elem = f64;

    /// Allocate a vector list and a rank-N normal random filler.
    fn new(dim: usize, n: usize, rng: RngConfig) -> Result<Self, TensorRandError> {
        assert!(dim > 0, "HaarVectors::new: dim must be > 0");
        assert!(n > 0, "HaarVectors::new: n must be > 0");

        Self::try_with_mode(dim, n, rng, FillMode::Stateful)
    }

    #[inline]
    /// Refresh the dense buffer with rank-N normal random values and normalize rows.
    fn refresh(&mut self) {
        self.try_refresh_stateful()
            .expect("invalid stateful Haar-vector refresh")
    }
}

impl HaarVectors {
    fn try_with_mode(
        dim: usize,
        n: usize,
        rng: RngConfig,
        mode: FillMode,
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
        let filler = match mode {
            FillMode::Stateful => TensorRandFiller::try_new(kind, rng)?,
            FillMode::Indexed => TensorRandFiller::try_new_indexed(kind, rng)?,
        };
        Ok(Self {
            vl: VectorList::empty(dim, n),
            dim,
            n,
            filler,
            mode,
            workspace,
        })
    }

    /// Constructs schedule-independent Haar vectors for explicit scientific steps.
    pub fn try_new_indexed(dim: usize, n: usize, rng: RngConfig) -> Result<Self, TensorRandError> {
        Self::try_with_mode(dim, n, rng, FillMode::Indexed)
    }

    /// Refreshes schedule-independent Haar vectors for an explicit step and domain.
    pub fn try_refresh_at(&mut self, step: u64, domain: u64) -> Result<(), TensorRandError> {
        if self.mode != FillMode::Indexed {
            return Err(TensorRandError::StatefulMethodDoesNotSupportIndexedFill);
        }
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

    fn try_refresh_stateful(&mut self) -> Result<(), TensorRandError> {
        if self.mode != FillMode::Stateful {
            return Err(TensorRandError::IndexedStepRequired);
        }
        match &mut self.workspace {
            HaarWorkspace::Sign { codes } => {
                self.filler.try_refresh(codes)?;
                write_signs(codes.data(), self.vl.as_tensor_mut().data_mut());
            }
            HaarWorkspace::Plane { units } => {
                self.filler.try_fill_slice(units)?;
                write_plane(units, self.vl.as_tensor_mut().data_mut());
            }
            HaarWorkspace::Sphere { units } => {
                self.filler.try_fill_slice(units)?;
                write_sphere(units, self.vl.as_tensor_mut().data_mut());
            }
            HaarWorkspace::Gaussian => {
                self.filler
                    .try_fill_slice(self.vl.as_tensor_mut().data_mut())?;
                self.vl.normalize();
            }
        }
        Ok(())
    }

    /// Returns the fully resolved random configuration used by this generator.
    pub fn rng_config(&self) -> RngConfig {
        self.filler.rng_config()
    }

    /// Build a Haar generator around existing `[n, dim]` vector-list data.
    ///
    /// The random filler is initialized with the standard-normal distribution
    /// used by `refresh`, so callers can import data and later resume random
    /// Haar generation with the same object.
    #[inline]
    pub fn from_ndarray(array: &Array2<f64>, rng: RngConfig) -> Result<Self, TensorRandError> {
        let mut generator =
            Self::try_with_mode(array.ncols(), array.nrows(), rng, FillMode::Stateful)?;
        generator.vl = VectorList::from_ndarray(array);
        Ok(generator)
    }

    /// Export inner vector-list storage to ndarray with shape `[n, dim]`.
    #[inline]
    pub fn to_ndarray(&self) -> Array2<f64> {
        self.vl.to_ndarray()
    }

    #[inline]
    /// Convert this Haar vector batch into a structured JSON value.
    pub fn serialize_value(&self) -> Result<Value, serde_json::Error> {
        self.vl.serialize_value()
    }

    #[inline]
    /// Convert this Haar vector batch into pretty JSON text.
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.vl.serialize()
    }
}

fn write_signs(codes: &[usize], output: &mut [f64]) {
    output
        .par_iter_mut()
        .zip(codes.par_iter())
        .for_each(|(value, code)| *value = if *code == 0 { -1.0 } else { 1.0 });
}

fn write_plane(units: &[f64], output: &mut [f64]) {
    output
        .par_chunks_exact_mut(2)
        .zip(units.par_iter())
        .for_each(|(row, unit)| {
            let (sin, cos) = (std::f64::consts::TAU * unit).sin_cos();
            row[0] = cos;
            row[1] = sin;
        });
}

fn write_sphere(units: &[f64], output: &mut [f64]) {
    output
        .par_chunks_exact_mut(3)
        .zip(units.par_chunks_exact(2))
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
    pub n: usize,
    code_buf: Tensor<usize>,       // shape [n], holds codes in [0, 2*dim)
    code_filler: TensorRandFiller, // RandType::UniformInt over code range
    mode: FillMode,
}

impl VectorListRand for NNVectors {
    type Elem = isize;

    /// Allocate output rows and a rank-N integer-code random filler.
    fn new(dim: usize, n: usize, rng: RngConfig) -> Result<Self, TensorRandError> {
        assert!(dim > 0, "NNVectors::new: dim must be > 0");
        assert!(n > 0, "NNVectors::new: n must be > 0");

        let vl = VectorList::<isize>::empty(dim, n);
        let code_buf = Tensor::<usize>::empty(vec![n].as_slice());

        let code_filler = TensorRandFiller::try_new(
            RandType::UniformInt {
                low: 0,
                high: (2 * dim) as i64 - 1,
            },
            rng,
        )?;

        Ok(Self {
            vl,
            dim,
            n,
            code_buf,
            code_filler,
            mode: FillMode::Stateful,
        })
    }

    #[inline]
    /// Refresh rank-N integer codes and decode them into vector-list rows.
    fn refresh(&mut self) {
        self.code_filler.refresh(&mut self.code_buf);
        decode_nearest_neighbor_codes(
            self.code_buf.data(),
            self.dim,
            self.vl.as_tensor_mut().data_mut(),
        );
    }
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
        .for_each(|(row, code)| decode_nearest_neighbor_code(*code, row));
}

impl NNVectors {
    /// Constructs schedule-independent nearest-neighbor vectors.
    pub fn try_new_indexed(dim: usize, n: usize, rng: RngConfig) -> Result<Self, TensorRandError> {
        assert!(dim > 0, "NNVectors::try_new_indexed: dim must be > 0");
        assert!(n > 0, "NNVectors::try_new_indexed: n must be > 0");
        Ok(Self {
            vl: VectorList::empty(dim, n),
            dim,
            n,
            code_buf: Tensor::empty(&[n]),
            code_filler: TensorRandFiller::try_new_indexed(
                RandType::UniformInt {
                    low: 0,
                    high: (2 * dim) as i64 - 1,
                },
                rng,
            )?,
            mode: FillMode::Indexed,
        })
    }

    /// Refreshes schedule-independent nearest-neighbor vectors.
    pub fn try_refresh_at(&mut self, step: u64, domain: u64) -> Result<(), TensorRandError> {
        if self.mode != FillMode::Indexed {
            return Err(TensorRandError::StatefulMethodDoesNotSupportIndexedFill);
        }
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

    /// Returns the fully resolved random configuration used by this generator.
    pub fn rng_config(&self) -> RngConfig {
        self.code_filler.rng_config()
    }

    /// Build a nearest-neighbor generator around existing `[n, dim]` rows.
    ///
    /// The integer-code filler is initialized with the same direction-code
    /// range used by `refresh`, so imported data can later be replaced by new
    /// random nearest-neighbor directions.
    #[inline]
    pub fn from_ndarray(array: &Array2<isize>, rng: RngConfig) -> Result<Self, TensorRandError> {
        let vl = VectorList::<isize>::from_ndarray(array);
        let dim = vl.dim();
        let n = vl.num_vectors();
        let code_buf = Tensor::<usize>::empty(vec![n].as_slice());
        let code_filler = TensorRandFiller::try_new(
            RandType::UniformInt {
                low: 0,
                high: (2 * dim) as i64 - 1,
            },
            rng,
        )?;
        Ok(Self {
            vl,
            dim,
            n,
            code_buf,
            code_filler,
            mode: FillMode::Stateful,
        })
    }

    /// Export inner vector-list storage to ndarray with shape `[n, dim]`.
    #[inline]
    pub fn to_ndarray(&self) -> Array2<isize> {
        self.vl.to_ndarray()
    }

    #[inline]
    /// Convert this nearest-neighbor vector batch into a structured JSON value.
    pub fn serialize_value(&self) -> Result<Value, serde_json::Error> {
        self.vl.serialize_value()
    }

    #[inline]
    /// Convert this nearest-neighbor vector batch into pretty JSON text.
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.vl.serialize()
    }
}

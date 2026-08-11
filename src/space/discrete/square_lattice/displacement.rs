/*!
Reproducible source-target pair generation for square lattices.

One call to [`RandPairGenerator::refresh_at`] deterministically fills a complete
pair batch from an explicit random key and sweep. The result is independent of
Rayon worker count and scheduling because each value has its own indexed random
coordinate.

Sources are valid lattice indices. Displacements and targets remain raw;
`SquareLattice` applies its configured boundary condition when those targets
are used.
*/

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::math::prelude::VectorList;

use super::kernel::{Kernel, KernelError, KernelType, try_create_kernel};
use super::random::{
    DOMAIN_HAAR_COMPONENT, DOMAIN_SOURCE_COORDINATE, RandomKey, standard_normal, uniform_index,
};

/// Rule for selecting each pair's source coordinate.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceMode {
    Origin,
    RandomUniform,
}

/// Invalid pair-generator construction.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum PairGenerationError {
    EmptyShape,
    ZeroAxis {
        axis: usize,
    },
    ZeroPairs,
    KernelRankMismatch {
        kernel_dimension: usize,
        rank: usize,
    },
    Kernel(KernelError),
}

impl fmt::Display for PairGenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EmptyShape => write!(
                formatter,
                "pair-generator shape must have at least one axis"
            ),
            Self::ZeroAxis { axis } => {
                write!(
                    formatter,
                    "pair-generator shape axis {axis} must be nonzero"
                )
            }
            Self::ZeroPairs => write!(formatter, "pair-generator pair count must be positive"),
            Self::KernelRankMismatch {
                kernel_dimension,
                rank,
            } => write!(
                formatter,
                "nearest-neighbor kernel dimension {kernel_dimension} does not match lattice rank {rank}"
            ),
            Self::Kernel(error) => error.fmt(formatter),
        }
    }
}

impl Error for PairGenerationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            _ => None,
        }
    }
}

impl From<KernelError> for PairGenerationError {
    fn from(value: KernelError) -> Self {
        Self::Kernel(value)
    }
}

/// Reusable buffers and immutable configuration for indexed random pairs.
#[derive(Clone)]
pub struct RandPairGenerator {
    shape: Vec<usize>,
    kernel: Box<dyn Kernel>,
    kernel_type: KernelType,
    source_mode: SourceMode,
    random_key: RandomKey,
    generated_sweep: Option<u64>,
    source_coords_cache: VectorList<isize>,
    direction_cache: Option<VectorList<f64>>,
    displacement_coords_cache: VectorList<isize>,
    target_coords_cache: VectorList<isize>,
}

impl fmt::Debug for RandPairGenerator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RandPairGenerator")
            .field("shape", &self.shape)
            .field("kernel_type", &self.kernel_type)
            .field("source_mode", &self.source_mode)
            .field("random_key", &self.random_key)
            .field("generated_sweep", &self.generated_sweep)
            .field("num_pairs", &self.num_pairs())
            .finish_non_exhaustive()
    }
}

impl RandPairGenerator {
    /// Constructs an allocation-stable pair generator from explicit randomness.
    pub fn new(
        shape: &[usize],
        kernel_type: KernelType,
        num_pairs: usize,
        source_mode: SourceMode,
        random_key: RandomKey,
    ) -> Result<Self, PairGenerationError> {
        validate_shape(shape)?;
        if num_pairs == 0 {
            return Err(PairGenerationError::ZeroPairs);
        }
        validate_kernel_rank(kernel_type, shape.len())?;
        let kernel = try_create_kernel(kernel_type)?;
        let rank = shape.len();
        let direction_cache = match kernel_type {
            KernelType::NearestNeighbor { .. } => None,
            KernelType::PowerLaw { .. } | KernelType::Uniform { .. } => {
                Some(VectorList::empty(rank, num_pairs))
            }
        };

        Ok(Self {
            shape: shape.to_vec(),
            kernel,
            kernel_type,
            source_mode,
            random_key,
            generated_sweep: None,
            source_coords_cache: VectorList::empty(rank, num_pairs),
            direction_cache,
            displacement_coords_cache: VectorList::empty(rank, num_pairs),
            target_coords_cache: VectorList::empty(rank, num_pairs),
        })
    }

    /// Replaces every cached pair for an explicit scientific sweep.
    ///
    /// Repeating this call with the same sweep reproduces the same batch
    /// exactly. No mutable RNG cursor participates in the result.
    pub fn refresh_at(&mut self, sweep: u64) {
        self.refresh_sources_at(sweep);
        self.refresh_displacements_at(sweep);
        self.refresh_targets();
        self.generated_sweep = Some(sweep);
    }

    fn refresh_sources_at(&mut self, sweep: u64) {
        match self.source_mode {
            SourceMode::Origin => self.source_coords_cache.fill(0),
            SourceMode::RandomUniform => {
                let shape = &self.shape;
                let key = self.random_key;
                self.source_coords_cache
                    .par_for_each_vec_mut(|pair_index, row| {
                        for (axis, coordinate) in row.iter_mut().enumerate() {
                            *coordinate = uniform_index(
                                key,
                                sweep,
                                DOMAIN_SOURCE_COORDINATE,
                                pair_index as u64,
                                axis as u64,
                                shape[axis],
                            ) as isize;
                        }
                    });
            }
        }
    }

    fn refresh_displacements_at(&mut self, sweep: u64) {
        match self.kernel_type {
            KernelType::NearestNeighbor { .. } => {
                let key = self.random_key;
                let kernel = &self.kernel;
                self.displacement_coords_cache
                    .par_for_each_vec_mut(|pair_index, row| {
                        let code = kernel.sample_indexed(key, sweep, pair_index as u64) as usize;
                        decode_nearest_neighbor_code(code, row);
                    });
            }
            KernelType::PowerLaw { .. } | KernelType::Uniform { .. } => {
                let key = self.random_key;
                let directions = self
                    .direction_cache
                    .as_mut()
                    .expect("non-nearest kernel has a direction cache");
                directions.par_for_each_vec_mut(|pair_index, row| {
                    let mut squared_norm = 0.0;
                    for (axis, component) in row.iter_mut().enumerate() {
                        let value = standard_normal(
                            key,
                            sweep,
                            DOMAIN_HAAR_COMPONENT,
                            pair_index as u64,
                            axis as u64,
                        );
                        *component = value;
                        squared_norm += value * value;
                    }
                    if squared_norm > 0.0 && squared_norm.is_finite() {
                        let inverse_norm = squared_norm.sqrt().recip();
                        for component in row {
                            *component *= inverse_norm;
                        }
                    } else {
                        row.fill(0.0);
                        row[0] = 1.0;
                    }
                });

                let directions = self
                    .direction_cache
                    .as_ref()
                    .expect("non-nearest kernel has a direction cache");
                let kernel = &self.kernel;
                self.displacement_coords_cache
                    .par_for_each_vec_mut(|pair_index, row| {
                        let length = kernel.sample_indexed(key, sweep, pair_index as u64);
                        for (axis, component) in row.iter_mut().enumerate() {
                            *component = (directions.get(pair_index as isize, axis as isize)
                                * length) as isize;
                        }
                    });
            }
        }
    }

    fn refresh_targets(&mut self) {
        let sources = &self.source_coords_cache;
        let displacements = &self.displacement_coords_cache;
        self.target_coords_cache
            .par_for_each_vec_mut(|pair_index, row| {
                for (axis, target) in row.iter_mut().enumerate() {
                    *target = sources.get(pair_index as isize, axis as isize)
                        + displacements.get(pair_index as isize, axis as isize);
                }
            });
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn num_pairs(&self) -> usize {
        self.source_coords_cache.num_vectors()
    }

    pub fn kernel_type(&self) -> KernelType {
        self.kernel_type
    }

    pub fn source_mode(&self) -> SourceMode {
        self.source_mode
    }

    pub fn random_key(&self) -> RandomKey {
        self.random_key
    }

    /// Returns the sweep represented by the current buffers, if generated.
    pub fn generated_sweep(&self) -> Option<u64> {
        self.generated_sweep
    }

    pub fn sources(&self) -> &VectorList<isize> {
        &self.source_coords_cache
    }

    pub fn displacements(&self) -> &VectorList<isize> {
        &self.displacement_coords_cache
    }

    pub fn targets(&self) -> &VectorList<isize> {
        &self.target_coords_cache
    }

    pub fn source(&self, index: isize) -> &[isize] {
        self.source_coords_cache.get_vec(index)
    }

    pub fn displacement(&self, index: isize) -> &[isize] {
        self.displacement_coords_cache.get_vec(index)
    }

    pub fn target(&self, index: isize) -> &[isize] {
        self.target_coords_cache.get_vec(index)
    }
}

fn validate_shape(shape: &[usize]) -> Result<(), PairGenerationError> {
    if shape.is_empty() {
        return Err(PairGenerationError::EmptyShape);
    }
    if let Some(axis) = shape.iter().position(|length| *length == 0) {
        return Err(PairGenerationError::ZeroAxis { axis });
    }
    Ok(())
}

fn validate_kernel_rank(kernel_type: KernelType, rank: usize) -> Result<(), PairGenerationError> {
    if let KernelType::NearestNeighbor { d } = kernel_type
        && d != rank
    {
        return Err(PairGenerationError::KernelRankMismatch {
            kernel_dimension: d,
            rank,
        });
    }
    Ok(())
}

fn decode_nearest_neighbor_code(code: usize, row: &mut [isize]) {
    let axis = code / 2;
    let sign = if code.is_multiple_of(2) { 1 } else { -1 };
    for (index, component) in row.iter_mut().enumerate() {
        *component = if index == axis { sign } else { 0 };
    }
}

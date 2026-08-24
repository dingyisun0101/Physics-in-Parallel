/*!
Reproducible source-target pair generation for square lattices.

One call to [`PairGenerator::refresh_at`] deterministically fills a complete
pair batch from an explicit random key and sweep. The result is independent of
Rayon worker count and scheduling because each value has its own indexed random
coordinate.

Independent-uniform pairing samples source and target sites directly and with
replacement. Kernel pairing samples a source and raw displacement. Both methods
produce the same buffers: sources are canonical lattice coordinates,
`target = source + displacement`, and targets remain raw coordinates for
`SquareLattice` to interpret through its configured boundary condition.
*/

use std::error::Error;
use std::fmt;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::math::tensor::rank_2::vector_list::{HaarVectors, NNVectors, VectorList};
use crate::math::tensor::rank_n::dense::Tensor as DenseTensor;
use crate::math::tensor::rank_n::layout::RowMajorLayout;
use crate::math::tensor::rank_n::tensor_trait::TensorTrait;
use crate::math::{RandType, TensorError, TensorRandError, TensorRandFiller};
use crate::threading::parallel_chunk_len;

use super::kernel::{BuiltinKernel, KernelError, KernelType, try_create_builtin_kernel};
use super::random::{
    DOMAIN_HAAR_COMPONENT, DOMAIN_INDEPENDENT_SOURCE_SITE, DOMAIN_INDEPENDENT_TARGET_SITE,
    DOMAIN_SOURCE_COORDINATE,
};
use crate::rng::{RngConfig, RngConfigError};

/// Rule for selecting each pair's source coordinate.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceMode {
    Origin,
    RandomUniform,
}

/// Complete rule used to generate each source-target pair.
///
/// `IndependentUniform` is deliberately not a displacement kernel: it samples
/// both endpoints independently and uniformly from all sites, with replacement.
/// Consequently, self-pairs are valid and require no exceptional path.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PairingMethod {
    IndependentUniform,
    Kernel {
        kernel: KernelType,
        sources: SourceMode,
    },
}

/// Immutable public configuration for an allocation-stable pair generator.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairGeneratorConfig {
    method: PairingMethod,
    num_pairs: usize,
    rng: RngConfig,
}

impl PairGeneratorConfig {
    pub const fn new(method: PairingMethod, num_pairs: usize, rng: RngConfig) -> Self {
        Self {
            method,
            num_pairs,
            rng,
        }
    }

    pub const fn independent_uniform(num_pairs: usize, rng: RngConfig) -> Self {
        Self::new(PairingMethod::IndependentUniform, num_pairs, rng)
    }

    pub const fn kernel(
        kernel: KernelType,
        num_pairs: usize,
        sources: SourceMode,
        rng: RngConfig,
    ) -> Self {
        Self::new(PairingMethod::Kernel { kernel, sources }, num_pairs, rng)
    }

    pub const fn method(self) -> PairingMethod {
        self.method
    }

    pub const fn num_pairs(self) -> usize {
        self.num_pairs
    }

    pub const fn rng(self) -> RngConfig {
        self.rng
    }
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
    SiteCountOverflow,
    PairBufferOverflow {
        num_pairs: usize,
        rank: usize,
    },
    KernelRankMismatch {
        kernel_dimension: usize,
        rank: usize,
    },
    Kernel(KernelError),
    RngConfig(RngConfigError),
    TensorRand(TensorRandError),
}

impl fmt::Display for PairGenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Self::SiteCountOverflow => {
                write!(
                    formatter,
                    "pair-generator shape exceeds the signed site-index space"
                )
            }
            Self::PairBufferOverflow { num_pairs, rank } => write!(
                formatter,
                "pair-generator buffer shape {num_pairs}x{rank} exceeds the signed index space"
            ),
            Self::KernelRankMismatch {
                kernel_dimension,
                rank,
            } => write!(
                formatter,
                "nearest-neighbor kernel dimension {kernel_dimension} does not match lattice rank {rank}"
            ),
            Self::Kernel(error) => error.fmt(formatter),
            Self::RngConfig(error) => error.fmt(formatter),
            Self::TensorRand(error) => error.fmt(formatter),
        }
    }
}

impl Error for PairGenerationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            Self::RngConfig(error) => Some(error),
            Self::TensorRand(error) => Some(error),
            _ => None,
        }
    }
}

impl From<KernelError> for PairGenerationError {
    fn from(value: KernelError) -> Self {
        Self::Kernel(value)
    }
}

impl From<RngConfigError> for PairGenerationError {
    fn from(value: RngConfigError) -> Self {
        Self::RngConfig(value)
    }
}

impl From<TensorRandError> for PairGenerationError {
    fn from(value: TensorRandError) -> Self {
        Self::TensorRand(value)
    }
}

/// Reusable buffers and immutable configuration for indexed random pairs.
#[derive(Clone)]
pub struct PairGenerator {
    layout: RowMajorLayout,
    config: PairGeneratorConfig,
    kernel: Option<BuiltinKernel>,
    random_filler: TensorRandFiller,
    workspace: PairingWorkspace,
    generated_sweep: Option<u64>,
    source_coords_cache: VectorList<isize>,
    displacement_coords_cache: VectorList<isize>,
    target_coords_cache: VectorList<isize>,
}

#[derive(Clone)]
enum PairingWorkspace {
    IndependentUniform,
    Kernel {
        source_sites: Option<DenseTensor<usize>>,
        directions: Box<KernelDirections>,
        length_units: Option<Vec<f64>>,
    },
}

#[derive(Clone)]
enum KernelDirections {
    NearestNeighbor(NNVectors),
    Radial(HaarVectors),
}

impl fmt::Debug for PairGenerator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairGenerator")
            .field("shape", &self.layout.shape())
            .field("method", &self.config.method())
            .field("rng", &self.random_filler.rng_config())
            .field("generated_sweep", &self.generated_sweep)
            .field("num_pairs", &self.num_pairs())
            .finish_non_exhaustive()
    }
}

impl PairGenerator {
    /// Constructs an allocation-stable pair generator from explicit randomness.
    pub fn new(shape: &[usize], config: PairGeneratorConfig) -> Result<Self, PairGenerationError> {
        let layout = validate_layout(shape)?;
        if config.num_pairs() == 0 {
            return Err(PairGenerationError::ZeroPairs);
        }

        let rank = shape.len();
        if config.num_pairs() > isize::MAX as usize
            || config
                .num_pairs()
                .checked_mul(rank)
                .is_none_or(|size| size > isize::MAX as usize)
        {
            return Err(PairGenerationError::PairBufferOverflow {
                num_pairs: config.num_pairs(),
                rank,
            });
        }
        let random_filler = TensorRandFiller::try_new_indexed(
            RandType::Uniform {
                low: 0.0,
                high: 1.0,
            },
            config.rng(),
        )?;
        let resolved_rng = random_filler.rng_config();
        let (kernel, workspace) = match config.method() {
            PairingMethod::IndependentUniform => (None, PairingWorkspace::IndependentUniform),
            PairingMethod::Kernel { kernel, sources } => {
                validate_kernel_rank(kernel, rank)?;
                let (directions, length_units) = match kernel {
                    KernelType::NearestNeighbor { .. } => (
                        KernelDirections::NearestNeighbor(NNVectors::try_new_indexed(
                            rank,
                            config.num_pairs(),
                            resolved_rng,
                        )?),
                        None,
                    ),
                    KernelType::PowerLaw { .. } | KernelType::UniformDistance { .. } => (
                        KernelDirections::Radial(HaarVectors::try_new_indexed(
                            rank,
                            config.num_pairs(),
                            resolved_rng,
                        )?),
                        Some(vec![0.0; config.num_pairs()]),
                    ),
                };
                (
                    Some(try_create_builtin_kernel(kernel)?),
                    PairingWorkspace::Kernel {
                        source_sites: (sources == SourceMode::RandomUniform)
                            .then(|| DenseTensor::empty(&[config.num_pairs()])),
                        directions: Box::new(directions),
                        length_units,
                    },
                )
            }
        };

        Ok(Self {
            layout,
            config,
            kernel,
            random_filler,
            workspace,
            generated_sweep: None,
            source_coords_cache: VectorList::empty(rank, config.num_pairs()),
            displacement_coords_cache: VectorList::empty(rank, config.num_pairs()),
            target_coords_cache: VectorList::empty(rank, config.num_pairs()),
        })
    }

    /// Replaces every cached pair for an explicit scientific sweep.
    ///
    /// Repeating this call with the same sweep reproduces the same batch
    /// exactly. No mutable RNG cursor participates in the result.
    pub fn refresh_at(&mut self, sweep: u64) {
        match &mut self.workspace {
            PairingWorkspace::IndependentUniform => {
                let rank = self.layout.shape().len();
                self.random_filler
                    .try_fill_index_pairs_with(
                        (
                            self.source_coords_cache.as_tensor_mut().data_mut(),
                            self.target_coords_cache.as_tensor_mut().data_mut(),
                            self.displacement_coords_cache.as_tensor_mut().data_mut(),
                        ),
                        rank,
                        self.layout.size(),
                        sweep,
                        (
                            DOMAIN_INDEPENDENT_SOURCE_SITE,
                            DOMAIN_INDEPENDENT_TARGET_SITE,
                        ),
                        |source_site, target_site, source, target, displacement| {
                            self.layout.coordinate_into(source_site, source);
                            self.layout.coordinate_into(target_site, target);
                            for ((component, &target), &source) in displacement
                                .iter_mut()
                                .zip(target.iter())
                                .zip(source.iter())
                            {
                                *component = target - source;
                            }
                        },
                    )
                    .expect("validated indexed independent-uniform filler");
            }
            PairingWorkspace::Kernel {
                source_sites,
                directions,
                length_units,
            } => {
                refresh_kernel_sources(
                    &self.layout,
                    &mut self.random_filler,
                    source_sites,
                    sweep,
                    &mut self.source_coords_cache,
                );
                match directions.as_mut() {
                    KernelDirections::NearestNeighbor(generator) => {
                        generator
                            .try_refresh_at(sweep, DOMAIN_HAAR_COMPONENT)
                            .expect("validated indexed nearest-neighbor generator");
                        assemble_nearest_neighbor(
                            &self.source_coords_cache,
                            &generator.vl,
                            &mut self.displacement_coords_cache,
                            &mut self.target_coords_cache,
                        );
                    }
                    KernelDirections::Radial(generator) => {
                        generator
                            .try_refresh_at(sweep, DOMAIN_HAAR_COMPONENT)
                            .expect("validated indexed Haar-vector generator");
                        let units = length_units
                            .as_mut()
                            .expect("radial kernel has a length-unit workspace");
                        self.random_filler
                            .try_fill_slice_at(units, sweep, super::random::DOMAIN_KERNEL_SAMPLE)
                            .expect("validated indexed kernel-length filler");
                        assemble_radial(
                            &self.source_coords_cache,
                            &generator.vl,
                            units,
                            self.kernel
                                .as_ref()
                                .expect("kernel method has a validated kernel"),
                            &mut self.displacement_coords_cache,
                            &mut self.target_coords_cache,
                        );
                    }
                }
            }
        }
        self.generated_sweep = Some(sweep);
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn rank(&self) -> usize {
        self.layout.shape().len()
    }

    pub fn num_pairs(&self) -> usize {
        self.source_coords_cache.num_vectors()
    }

    pub fn method(&self) -> PairingMethod {
        self.config.method()
    }

    pub fn config(&self) -> PairGeneratorConfig {
        PairGeneratorConfig::new(
            self.config.method(),
            self.config.num_pairs(),
            self.rng_config(),
        )
    }

    pub fn rng_config(&self) -> RngConfig {
        self.random_filler.rng_config()
    }

    /// Returns the sweep represented by the current buffers, if generated.
    pub fn generated_sweep(&self) -> Option<u64> {
        self.generated_sweep
    }

    pub fn sources(&self) -> &VectorList<isize> {
        &self.source_coords_cache
    }

    /// Raw displacements. Every row satisfies `target = source + displacement`.
    pub fn displacements(&self) -> &VectorList<isize> {
        &self.displacement_coords_cache
    }

    /// Raw target coordinates for use with boundary-aware `SquareLattice` accessors.
    pub fn targets(&self) -> &VectorList<isize> {
        &self.target_coords_cache
    }

    pub fn source(&self, index: isize) -> &[isize] {
        self.source_coords_cache.vector(index)
    }

    /// Returns the raw displacement for one pair.
    pub fn displacement(&self, index: isize) -> &[isize] {
        self.displacement_coords_cache.vector(index)
    }

    /// Returns a raw target coordinate. Pass it to boundary-aware space accessors.
    pub fn target(&self, index: isize) -> &[isize] {
        self.target_coords_cache.vector(index)
    }
}

fn validate_layout(shape: &[usize]) -> Result<RowMajorLayout, PairGenerationError> {
    RowMajorLayout::try_new(shape).map_err(|error| match error {
        TensorError::InvalidShape { shape } if shape.is_empty() => PairGenerationError::EmptyShape,
        TensorError::InvalidShape { shape } => PairGenerationError::ZeroAxis {
            axis: shape
                .iter()
                .position(|extent| *extent == 0)
                .expect("invalid nonempty tensor shape has a zero axis"),
        },
        TensorError::ShapeProductOverflow { .. } | TensorError::IndexSpaceOverflow { .. } => {
            PairGenerationError::SiteCountOverflow
        }
        _ => unreachable!("layout construction reports only shape errors"),
    })
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

fn refresh_kernel_sources(
    layout: &RowMajorLayout,
    filler: &mut TensorRandFiller,
    sites: &mut Option<DenseTensor<usize>>,
    sweep: u64,
    sources: &mut VectorList<isize>,
) {
    let Some(sites) = sites else {
        return;
    };
    filler
        .try_fill_indices_at(
            sites.data_mut(),
            layout.size(),
            sweep,
            DOMAIN_SOURCE_COORDINATE,
        )
        .expect("validated indexed source-site filler");
    let rank = layout.shape().len();
    let min_pairs_per_job = parallel_chunk_len(sources.num_vectors()).unwrap_or(1);
    sources
        .as_tensor_mut()
        .data_mut()
        .par_chunks_exact_mut(rank)
        .zip(sites.data().par_iter())
        .with_min_len(min_pairs_per_job)
        .for_each(|(row, site)| layout.coordinate_into(*site, row));
}

fn assemble_nearest_neighbor(
    sources: &VectorList<isize>,
    directions: &VectorList<isize>,
    displacements: &mut VectorList<isize>,
    targets: &mut VectorList<isize>,
) {
    let rank = sources.dim();
    let min_pairs_per_job = parallel_chunk_len(sources.num_vectors()).unwrap_or(1);
    sources
        .as_tensor()
        .data()
        .par_chunks_exact(rank)
        .zip(directions.as_tensor().data().par_chunks_exact(rank))
        .zip(
            displacements
                .as_tensor_mut()
                .data_mut()
                .par_chunks_exact_mut(rank),
        )
        .zip(
            targets
                .as_tensor_mut()
                .data_mut()
                .par_chunks_exact_mut(rank),
        )
        .with_min_len(min_pairs_per_job)
        .for_each(|(((source, direction), displacement), target)| {
            for (((displacement, target), &source), &direction) in displacement
                .iter_mut()
                .zip(target.iter_mut())
                .zip(source.iter())
                .zip(direction.iter())
            {
                *displacement = direction;
                *target = source + direction;
            }
        });
}

fn assemble_radial(
    sources: &VectorList<isize>,
    directions: &VectorList<f64>,
    length_units: &[f64],
    kernel: &BuiltinKernel,
    displacements: &mut VectorList<isize>,
    targets: &mut VectorList<isize>,
) {
    let rank = sources.dim();
    let min_pairs_per_job = parallel_chunk_len(sources.num_vectors()).unwrap_or(1);
    sources
        .as_tensor()
        .data()
        .par_chunks_exact(rank)
        .zip(directions.as_tensor().data().par_chunks_exact(rank))
        .zip(length_units.par_iter())
        .zip(
            displacements
                .as_tensor_mut()
                .data_mut()
                .par_chunks_exact_mut(rank),
        )
        .zip(
            targets
                .as_tensor_mut()
                .data_mut()
                .par_chunks_exact_mut(rank),
        )
        .with_min_len(min_pairs_per_job)
        .for_each(|((((source, direction), unit), displacement), target)| {
            let length = kernel.sample_unit(*unit);
            for (((displacement, target), &source), &direction) in displacement
                .iter_mut()
                .zip(target.iter_mut())
                .zip(source.iter())
                .zip(direction.iter())
            {
                let component = (direction * length) as isize;
                *displacement = component;
                *target = source + component;
            }
        });
}

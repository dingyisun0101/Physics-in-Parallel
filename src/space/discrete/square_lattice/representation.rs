/*!
Square-lattice representation for discrete spaces.

Purpose:
    This module defines a square, cubic, or hypercubic lattice topology over a
    tensor-style shape such as `[128]`, `[64, 64]`, or `[32, 64, 16]`. The shape
    lists the number of lattice sites on each axis, exactly like a tensor shape.
    Future discrete-space families with different neighbor geometry should live
    in sibling modules with their own names and invariants.

Storage model:
    - SquareLattice sites are stored in a dense rank-N tensor with the requested
      shape.
    - SquareLattice code handles spatial semantics: boundary normalization, vacancy
      sentinels, shape validation, initialization, and point downsampling.
    - Tensor code handles dense storage, row-major indexing, parallel fill, and
      scalar type constraints.

Boundary conditions:
    - `Periodic`: coordinates wrap around the lattice, so `-1` selects the last
      site on that axis.
    - `Reflective`: coordinates reflect at the lattice walls, modelling a
      hard-wall mirror boundary for index lookup.
*/

use std::path::PathBuf;
use std::{error::Error, fmt};

use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

use crate::math::TensorError;
use crate::math::scalar::{Scalar, ScalarSerde};
use crate::math::tensor::rank_n::{Dense as DenseBackend, RowMajorLayout, Tensor};
use crate::sampling::shuffle_slice_indexed;
use crate::space::space_trait::Space;
use crate::threading::parallel_chunk_len;

use super::random::IndexedRng;
use crate::rng::{ResolvedRng, RngError};

const DOMAIN_LATTICE_INITIALIZATION: u64 = 0xc762_ba71_b5a7_8f31;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryCondition {
    Periodic,
    Reflective,
    /// Zero-normal-gradient boundary used by finite-difference operators.
    Neumann,
}

impl BoundaryCondition {
    #[inline]
    pub(crate) fn normalize(self, coord: isize, side_len: usize) -> usize {
        debug_assert!(side_len > 0);
        match self {
            Self::Periodic => wrap_periodic(coord, side_len),
            Self::Reflective => reflect_coordinate(coord, side_len),
            Self::Neumann => coord.clamp(0, side_len as isize - 1) as usize,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SquareLatticeConfig {
    shape: Vec<usize>,
    boundary: BoundaryCondition,
    spacing: Vec<f64>,
    #[serde(skip)]
    layout: RowMajorLayout,
}

impl SquareLatticeConfig {
    /// Creates a validated square-lattice configuration.
    pub fn try_new(
        shape: &[usize],
        boundary: BoundaryCondition,
        spacing: Option<&[f64]>,
    ) -> Result<Self, SquareLatticeConfigError> {
        if shape.is_empty() {
            return Err(SquareLatticeConfigError::EmptyShape);
        }
        let layout = RowMajorLayout::try_new(shape).map_err(|error| match error {
            TensorError::InvalidShape { .. } if shape.is_empty() => {
                SquareLatticeConfigError::EmptyShape
            }
            TensorError::InvalidShape { .. } => SquareLatticeConfigError::ZeroAxis {
                axis: shape
                    .iter()
                    .position(|extent| *extent == 0)
                    .expect("invalid nonempty shape has a zero axis"),
            },
            TensorError::ShapeProductOverflow { .. } => {
                SquareLatticeConfigError::SiteCountOverflow {
                    shape: shape.to_vec(),
                }
            }
            TensorError::IndexSpaceOverflow { .. } => SquareLatticeConfigError::SiteCountOverflow {
                shape: shape.to_vec(),
            },
            _ => unreachable!("row-major layout construction reports only shape errors"),
        })?;
        let spacing = match spacing {
            Some(spacing) if spacing.len() != shape.len() => {
                return Err(SquareLatticeConfigError::SpacingRank {
                    expected: shape.len(),
                    actual: spacing.len(),
                });
            }
            Some(spacing) => spacing.to_vec(),
            None => vec![1.0; shape.len()],
        };
        for (axis, value) in spacing.iter().copied().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(SquareLatticeConfigError::InvalidSpacing { axis });
            }
        }
        Ok(Self {
            shape: shape.to_vec(),
            boundary,
            spacing,
            layout,
        })
    }

    /// Creates a configuration, panicking if its shape is invalid.
    ///
    /// New fallible boundaries should prefer [`Self::try_new`].
    #[inline]
    pub(crate) fn new(
        shape: &[usize],
        boundary: BoundaryCondition,
        spacing: Option<&[f64]>,
    ) -> Self {
        Self::try_new(shape, boundary, spacing)
            .unwrap_or_else(|error| panic!("invalid SquareLatticeConfig: {error}"))
    }

    #[inline]
    pub fn periodic(shape: &[usize]) -> Self {
        Self::new(shape, BoundaryCondition::Periodic, None)
    }

    #[inline]
    pub fn reflective(shape: &[usize]) -> Self {
        Self::new(shape, BoundaryCondition::Reflective, None)
    }

    /// Returns the configured boundary condition.
    #[inline]
    pub const fn boundary(&self) -> BoundaryCondition {
        self.boundary
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn num_sites(&self) -> usize {
        self.layout.size()
    }

    #[inline]
    pub(crate) fn tensor_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns one physical spacing per lattice axis.
    #[inline]
    pub fn spacing(&self) -> &[f64] {
        &self.spacing
    }

    /// Resolves a signed coordinate to its row-major site index.
    ///
    /// Every axis follows this lattice's configured boundary condition.
    pub fn flat_index(&self, coordinate: &[isize]) -> usize {
        assert_eq!(
            coordinate.len(),
            self.rank(),
            "lattice coordinate rank mismatch: expected {}, got {}",
            self.rank(),
            coordinate.len()
        );
        coordinate
            .iter()
            .zip(self.shape.iter())
            .zip(self.layout.strides().iter())
            .map(|((&component, &extent), &stride)| {
                self.boundary.normalize(component, extent) * stride
            })
            .sum()
    }

    /// Converts a signed flat row-major site into its canonical coordinate.
    ///
    /// Flat indices follow the configured boundary condition over the complete
    /// linear site buffer, matching coordinate access without requiring caller
    /// normalization.
    pub fn coordinate(&self, flat: isize) -> Vec<isize> {
        let flat = self.boundary.normalize(flat, self.num_sites());
        self.layout
            .coordinate(flat)
            .expect("normalized flat lattice index is valid")
    }

    /// Resolves one axis neighbor under the configured boundary.
    pub fn neighbor(&self, flat: isize, axis: usize, offset: isize) -> Option<usize> {
        if axis >= self.rank() {
            return None;
        }
        let flat = self.boundary.normalize(flat, self.num_sites());
        Some(self.neighbor_canonical(flat, axis, offset))
    }

    #[inline]
    fn neighbor_canonical(&self, flat: usize, axis: usize, offset: isize) -> usize {
        let stride = self.layout.strides()[axis];
        let coordinate = (flat / stride) % self.shape[axis];
        let normalized = self
            .boundary
            .normalize(coordinate as isize + offset, self.shape[axis]);
        flat - coordinate * stride + normalized * stride
    }

    /// Applies the scalar-grid Laplacian independently to interleaved components.
    pub fn laplacian(
        &self,
        input: &[f64],
        components: usize,
        output: &mut [f64],
    ) -> Result<(), SquareLatticeConfigError> {
        let expected = self
            .num_sites()
            .checked_mul(components)
            .ok_or(SquareLatticeConfigError::ValueCountOverflow)?;
        if components == 0 || input.len() != expected || output.len() != expected {
            return Err(SquareLatticeConfigError::ValueLayout {
                expected,
                input: input.len(),
                output: output.len(),
            });
        }
        let inverse_spacing_squared: Vec<f64> = self
            .spacing
            .iter()
            .map(|spacing| 1.0 / (spacing * spacing))
            .collect();
        let min_sites_per_job = parallel_chunk_len(self.num_sites()).unwrap_or(1);
        output
            .par_chunks_mut(components)
            .with_min_len(min_sites_per_job)
            .enumerate()
            .for_each(|(flat, output_site)| {
                output_site.fill(0.0);
                let center_start = flat * components;
                for (axis, inverse_spacing_squared) in
                    inverse_spacing_squared.iter().copied().enumerate()
                {
                    let plus_start = self.neighbor_canonical(flat, axis, 1) * components;
                    let minus_start = self.neighbor_canonical(flat, axis, -1) * components;
                    for component in 0..components {
                        output_site[component] += (input[plus_start + component]
                            + input[minus_start + component]
                            - 2.0 * input[center_start + component])
                            * inverse_spacing_squared;
                    }
                }
            });
        Ok(())
    }
}

impl<'de> Deserialize<'de> for SquareLatticeConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Document {
            shape: Vec<usize>,
            boundary: BoundaryCondition,
            #[serde(default)]
            spacing: Option<Vec<f64>>,
        }

        let document = Document::deserialize(deserializer)?;
        Self::try_new(
            &document.shape,
            document.boundary,
            document.spacing.as_deref(),
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Invalid square-lattice geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SquareLatticeConfigError {
    /// A lattice must have at least one axis.
    EmptyShape,
    /// Every declared axis must contain at least one site.
    ZeroAxis {
        /// Zero-length axis index.
        axis: usize,
    },
    /// Multiplying axis lengths overflowed the platform site count.
    SiteCountOverflow {
        /// Rejected shape.
        shape: Vec<usize>,
    },
    /// Physical spacing count did not match the lattice rank.
    SpacingRank { expected: usize, actual: usize },
    /// One physical spacing was non-finite or not positive.
    InvalidSpacing { axis: usize },
    /// Interleaved component value count overflowed `usize`.
    ValueCountOverflow,
    /// Input or output did not match `num_sites * components`.
    ValueLayout {
        expected: usize,
        input: usize,
        output: usize,
    },
    /// Random choice initialization had no candidate values.
    EmptyChoices,
    /// Random weights did not match the number of choices.
    WeightCount { choices: usize, weights: usize },
    /// One random weight was negative or non-finite.
    InvalidWeight { index: usize },
    /// Every random weight was zero.
    ZeroWeight,
    /// Explicit lattice values did not match the site count.
    ValueCount { expected: usize, actual: usize },
    /// Unified RNG configuration was incompatible with indexed initialization.
    Rng(RngError),
}

impl fmt::Display for SquareLatticeConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape => formatter.write_str("square lattice shape must not be empty"),
            Self::ZeroAxis { axis } => {
                write!(formatter, "square lattice axis {axis} has zero length")
            }
            Self::SiteCountOverflow { shape } => {
                write!(formatter, "square lattice shape {shape:?} overflows usize")
            }
            Self::SpacingRank { expected, actual } => write!(
                formatter,
                "square lattice spacing rank mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidSpacing { axis } => write!(
                formatter,
                "square lattice spacing at axis {axis} must be finite and positive"
            ),
            Self::ValueCountOverflow => {
                formatter.write_str("square lattice component value count overflows usize")
            }
            Self::ValueLayout {
                expected,
                input,
                output,
            } => write!(
                formatter,
                "square lattice component layout requires {expected} values, got input {input} and output {output}"
            ),
            Self::EmptyChoices => {
                formatter.write_str("random lattice initialization requires at least one choice")
            }
            Self::WeightCount { choices, weights } => write!(
                formatter,
                "random lattice initialization has {choices} choices but {weights} weights"
            ),
            Self::InvalidWeight { index } => write!(
                formatter,
                "random lattice initialization weight {index} must be finite and nonnegative"
            ),
            Self::ZeroWeight => formatter
                .write_str("random lattice initialization requires at least one positive weight"),
            Self::ValueCount { expected, actual } => write!(
                formatter,
                "lattice initialization requires {expected} values, got {actual}"
            ),
            Self::Rng(error) => error.fmt(formatter),
        }
    }
}

impl Error for SquareLatticeConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Rng(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum SquareLatticeInitMethod<T: Scalar> {
    /// Fill every site with `T::default()`.
    Empty,
    Uniform {
        val: T,
    },
    /// Independently sample choices using optional relative weights.
    RandomChoices {
        choices: Vec<T>,
        weights: Option<Vec<f64>>,
        rng: ResolvedRng,
    },
    /// Use explicit row-major values.
    Values {
        values: Vec<T>,
    },
    /// Use explicit row-major values after one unbiased seeded permutation.
    ShuffledValues {
        values: Vec<T>,
        rng: ResolvedRng,
    },
    SeededCenter {
        val: T,
    },
}

#[derive(Debug, Clone)]
pub struct SquareLattice<T: Scalar> {
    cfg: SquareLatticeConfig,
    cells: Tensor<T, DenseBackend>,
    initialization_rng: Option<ResolvedRng>,
}

impl<T: Scalar> SquareLattice<T> {
    pub fn new(
        cfg: SquareLatticeConfig,
        init_method: SquareLatticeInitMethod<T>,
    ) -> Result<Self, SquareLatticeConfigError> {
        let mut lattice = Self {
            cells: Tensor::<T, DenseBackend>::empty(&cfg.tensor_shape()),
            cfg,
            initialization_rng: None,
        };

        match init_method {
            SquareLatticeInitMethod::Empty => {}
            SquareLatticeInitMethod::Uniform { val } => lattice.cells.fill(val),
            SquareLatticeInitMethod::RandomChoices {
                choices,
                weights,
                rng,
            } => {
                if choices.is_empty() {
                    return Err(SquareLatticeConfigError::EmptyChoices);
                }
                let key = IndexedRng::new(rng).map_err(SquareLatticeConfigError::Rng)?;
                lattice.initialization_rng = Some(key.resolved_rng());
                let cumulative = cumulative_weights(&choices, weights.as_deref())?;
                let min_sites_per_job = parallel_chunk_len(lattice.num_sites()).unwrap_or(1);
                lattice
                    .cells_mut()
                    .par_iter_mut()
                    .with_min_len(min_sites_per_job)
                    .enumerate()
                    .for_each(|(index, slot)| {
                        let selected = match &cumulative {
                            Some(cumulative) => {
                                let sample =
                                    key.unit_f64(
                                        0,
                                        DOMAIN_LATTICE_INITIALIZATION,
                                        index as u64,
                                        0,
                                        0,
                                    ) * cumulative.last().copied().expect("nonempty weights");
                                cumulative.partition_point(|weight| *weight <= sample)
                            }
                            None => key
                                .uniform_index(
                                    0,
                                    DOMAIN_LATTICE_INITIALIZATION,
                                    index as u64,
                                    0,
                                    choices.len(),
                                )
                                .expect("nonempty choices"),
                        };
                        *slot = choices[selected.min(choices.len() - 1)];
                    });
            }
            SquareLatticeInitMethod::Values { values } => {
                if values.len() != lattice.cfg.num_sites() {
                    return Err(SquareLatticeConfigError::ValueCount {
                        expected: lattice.cfg.num_sites(),
                        actual: values.len(),
                    });
                }
                lattice.cells =
                    Tensor::<T, DenseBackend>::from_vec(&lattice.cfg.tensor_shape(), values);
            }
            SquareLatticeInitMethod::ShuffledValues { mut values, rng } => {
                if values.len() != lattice.cfg.num_sites() {
                    return Err(SquareLatticeConfigError::ValueCount {
                        expected: lattice.cfg.num_sites(),
                        actual: values.len(),
                    });
                }
                lattice.initialization_rng = Some(
                    shuffle_slice_indexed(&mut values, rng)
                        .map_err(SquareLatticeConfigError::Rng)?,
                );
                lattice.cells =
                    Tensor::<T, DenseBackend>::from_vec(&lattice.cfg.tensor_shape(), values);
            }
            SquareLatticeInitMethod::SeededCenter { val } => {
                let center: Vec<isize> = lattice
                    .cfg
                    .shape
                    .iter()
                    .map(|&axis_len| (axis_len / 2) as isize)
                    .collect();
                lattice.cells.set(&center, val);
            }
        }

        Ok(lattice)
    }

    pub(crate) fn downsample(&self, target_shape: &[usize]) -> Self {
        assert_eq!(
            target_shape.len(),
            self.cfg.rank(),
            "downsample rank mismatch: expected {}, got {}",
            self.cfg.rank(),
            target_shape.len()
        );
        assert!(
            target_shape.iter().all(|&n| n > 0),
            "downsample target shape must have only nonzero axis lengths; got {target_shape:?}"
        );
        assert!(
            target_shape
                .iter()
                .zip(self.cfg.shape().iter())
                .all(|(&new_dim, &old_dim)| new_dim <= old_dim),
            "downsample target shape must not exceed source shape: source {:?}, target {target_shape:?}",
            self.cfg.shape()
        );
        if target_shape == self.cfg.shape() {
            return self.clone();
        }

        let d = self.cfg.rank();
        let scale: Vec<f64> = self
            .cfg
            .shape
            .iter()
            .zip(target_shape.iter())
            .map(|(&old_dim, &new_dim)| old_dim as f64 / new_dim as f64)
            .collect();
        let new_cfg = SquareLatticeConfig::new(target_shape, self.cfg.boundary(), None);
        let mut new = Self {
            cells: Tensor::<T, DenseBackend>::empty(&new_cfg.tensor_shape()),
            cfg: new_cfg,
            initialization_rng: None,
        };

        let min_sites_per_job = parallel_chunk_len(new.num_sites()).unwrap_or(1);
        new.cells_mut()
            .par_iter_mut()
            .with_min_len(min_sites_per_job)
            .enumerate()
            .for_each(|(flat, slot)| {
                let mut rem = flat;
                let mut coord_new = vec![0usize; d];
                for axis in (0..d).rev() {
                    coord_new[axis] = rem % target_shape[axis];
                    rem /= target_shape[axis];
                }

                let coord_old: Vec<isize> = coord_new
                    .iter()
                    .enumerate()
                    .map(|(axis, &x)| (x as f64 * scale[axis]).floor() as isize)
                    .collect();
                *slot = *self.get(&coord_old);
            });

        new
    }
}

impl<T: Scalar> SquareLattice<T> {
    /// Returns the authoritative geometry configuration.
    #[inline]
    pub fn config(&self) -> &SquareLatticeConfig {
        &self.cfg
    }

    /// Returns resolved RNG provenance when this lattice used random initialization.
    #[inline]
    pub const fn initialization_resolved_rng(&self) -> Option<ResolvedRng> {
        self.initialization_rng
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        self.cells.storage().data()
    }

    /// Returns the tensor-style lattice shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.cfg.shape()
    }

    /// Returns the number of spatial axes.
    #[inline]
    pub fn rank(&self) -> usize {
        self.cfg.rank()
    }

    /// Returns the total number of lattice sites.
    #[inline]
    pub fn num_sites(&self) -> usize {
        self.cfg.num_sites()
    }

    /// Borrows a site after applying the configured boundary condition.
    #[inline]
    pub fn get(&self, coord: &[isize]) -> &T {
        &self.data()[self.cfg.flat_index(coord)]
    }

    /// Mutably borrows a site after applying the configured boundary condition.
    #[inline]
    pub fn get_mut(&mut self, coord: &[isize]) -> &mut T {
        let flat = self.cfg.flat_index(coord);
        &mut self.cells_mut()[flat]
    }

    /// Stores one site after applying the configured boundary condition.
    #[inline]
    pub fn set(&mut self, coord: &[isize], value: T) {
        let flat = self.cfg.flat_index(coord);
        self.cells_mut()[flat] = value;
    }

    /// Borrows a site through a signed flat index under the configured boundary.
    #[inline]
    pub fn get_flat(&self, flat: isize) -> &T {
        let flat = self.cfg.boundary.normalize(flat, self.num_sites());
        &self.data()[flat]
    }

    /// Mutably borrows a site through a signed flat index under the configured boundary.
    #[inline]
    pub fn get_flat_mut(&mut self, flat: isize) -> &mut T {
        let flat = self.cfg.boundary.normalize(flat, self.num_sites());
        &mut self.cells_mut()[flat]
    }

    /// Stores one site through a signed flat index under the configured boundary.
    #[inline]
    pub fn set_flat(&mut self, flat: isize, value: T) {
        *self.get_flat_mut(flat) = value;
    }

    /// Fills every lattice site with one value.
    #[inline]
    pub fn fill(&mut self, value: T) {
        self.cells.fill(value);
    }

    #[inline]
    pub(crate) fn cells_mut(&mut self) -> &mut [T] {
        self.cells.storage_mut().data_mut()
    }

    #[inline]
    pub(crate) fn tensor_shape(&self) -> Vec<usize> {
        self.cfg.tensor_shape()
    }

    #[inline]
    pub(crate) fn from_parts(cfg: SquareLatticeConfig, data: Vec<T>) -> Self {
        let expected = cfg.num_sites();
        assert_eq!(
            data.len(),
            expected,
            "lattice data length mismatch: expected {expected}, got {}",
            data.len()
        );
        Self {
            cells: Tensor::<T, DenseBackend>::from_vec(&cfg.tensor_shape(), data),
            cfg,
            initialization_rng: None,
        }
    }
}

impl<T: ScalarSerde> Space<T> for SquareLattice<T> {
    #[inline]
    fn data(&self) -> &[T] {
        self.data()
    }

    #[inline]
    fn dims(&self) -> Vec<usize> {
        self.cfg.tensor_shape()
    }

    #[inline]
    fn linear_size(&self) -> usize {
        self.cfg.num_sites()
    }

    #[inline]
    fn get(&self, coord: &[isize]) -> &T {
        self.get(coord)
    }

    #[inline]
    fn get_mut(&mut self, coord: &[isize]) -> &mut T {
        self.get_mut(coord)
    }

    #[inline]
    fn set(&mut self, coord: &[isize], val: T) {
        self.set(coord, val);
    }

    #[inline]
    fn save(&self, output_file: &PathBuf, l_target: usize) -> std::io::Result<()> {
        let target_shape = vec![l_target; self.cfg.rank()];
        crate::space::io::square_lattice::save_square_lattice(self, &target_shape, output_file)
    }

    #[inline]
    fn set_all(&mut self, val: T) {
        self.fill(val);
    }
}

fn cumulative_weights<T>(
    choices: &[T],
    weights: Option<&[f64]>,
) -> Result<Option<Vec<f64>>, SquareLatticeConfigError> {
    let Some(weights) = weights else {
        return Ok(None);
    };
    if weights.len() != choices.len() {
        return Err(SquareLatticeConfigError::WeightCount {
            choices: choices.len(),
            weights: weights.len(),
        });
    }
    let mut total = 0.0;
    let mut cumulative = Vec::with_capacity(weights.len());
    for (index, weight) in weights.iter().copied().enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(SquareLatticeConfigError::InvalidWeight { index });
        }
        total += weight;
        cumulative.push(total);
    }
    if total == 0.0 {
        return Err(SquareLatticeConfigError::ZeroWeight);
    }
    Ok(Some(cumulative))
}

#[inline]
fn wrap_periodic(coord: isize, side_len: usize) -> usize {
    let side_len = side_len as isize;
    let mut wrapped = coord % side_len;
    if wrapped < 0 {
        wrapped += side_len;
    }
    wrapped as usize
}

#[inline]
fn reflect_coordinate(coord: isize, side_len: usize) -> usize {
    if side_len == 1 {
        return 0;
    }
    let period = 2 * (side_len as isize - 1);
    let mut reflected = coord % period;
    if reflected < 0 {
        reflected += period;
    }
    if reflected >= side_len as isize {
        (period - reflected) as usize
    } else {
        reflected as usize
    }
}

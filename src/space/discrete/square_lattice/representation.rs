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

use crate::math::prelude::{DenseBackend, Scalar, ScalarSerde, Tensor};
use crate::space::space_trait::Space;

use super::random::RandomKey;

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
    pub fn normalize(self, coord: isize, side_len: usize) -> usize {
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
    strides: Vec<usize>,
    #[serde(skip)]
    num_sites: usize,
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
        let mut sites = 1usize;
        for (axis, &length) in shape.iter().enumerate() {
            if length == 0 {
                return Err(SquareLatticeConfigError::ZeroAxis { axis });
            }
            sites = sites.checked_mul(length).ok_or_else(|| {
                SquareLatticeConfigError::SiteCountOverflow {
                    shape: shape.to_vec(),
                }
            })?;
        }
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
        let mut strides = vec![1; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }
        Ok(Self {
            shape: shape.to_vec(),
            boundary,
            spacing,
            strides,
            num_sites: sites,
        })
    }

    /// Creates a configuration, panicking if its shape is invalid.
    ///
    /// New fallible boundaries should prefer [`Self::try_new`].
    #[inline]
    pub fn new(shape: &[usize], boundary: BoundaryCondition, spacing: Option<&[f64]>) -> Self {
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
        self.num_sites
    }

    #[inline]
    pub fn tensor_shape(&self) -> Vec<usize> {
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

    /// Returns cached row-major strides.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Converts a valid flat row-major site into its coordinate.
    pub fn coordinate(&self, flat: usize) -> Option<Vec<usize>> {
        (flat < self.num_sites).then(|| {
            self.strides
                .iter()
                .zip(&self.shape)
                .map(|(&stride, &length)| (flat / stride) % length)
                .collect()
        })
    }

    /// Resolves one axis neighbor under the configured boundary.
    pub fn neighbor(&self, flat: usize, axis: usize, offset: isize) -> Option<usize> {
        if flat >= self.num_sites || axis >= self.rank() {
            return None;
        }
        let coordinate = (flat / self.strides[axis]) % self.shape[axis];
        let normalized = self
            .boundary
            .normalize(coordinate as isize + offset, self.shape[axis]);
        Some(flat - coordinate * self.strides[axis] + normalized * self.strides[axis])
    }

    /// Applies the scalar-grid Laplacian independently to interleaved components.
    pub fn laplacian(
        &self,
        input: &[f64],
        components: usize,
        output: &mut [f64],
    ) -> Result<(), SquareLatticeConfigError> {
        let expected = self
            .num_sites
            .checked_mul(components)
            .ok_or(SquareLatticeConfigError::ValueCountOverflow)?;
        if components == 0 || input.len() != expected || output.len() != expected {
            return Err(SquareLatticeConfigError::ValueLayout {
                expected,
                input: input.len(),
                output: output.len(),
            });
        }
        for flat in 0..self.num_sites {
            for component in 0..components {
                let center_index = flat * components + component;
                let center = input[center_index];
                let mut value = 0.0;
                for axis in 0..self.rank() {
                    let plus = self
                        .neighbor(flat, axis, 1)
                        .expect("validated site and axis");
                    let minus = self
                        .neighbor(flat, axis, -1)
                        .expect("validated site and axis");
                    value += (input[plus * components + component]
                        + input[minus * components + component]
                        - 2.0 * center)
                        / self.spacing[axis].powi(2);
                }
                output[center_index] = value;
            }
        }
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
        }
    }
}

impl Error for SquareLatticeConfigError {}

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
        key: RandomKey,
    },
    /// Use explicit row-major values.
    Values {
        values: Vec<T>,
    },
    SeededCenter {
        val: T,
    },
}

#[derive(Debug, Clone)]
pub struct SquareLattice<T: Scalar> {
    cfg: SquareLatticeConfig,
    cells: Tensor<T, DenseBackend>,
}

impl<T: Scalar> SquareLattice<T> {
    pub fn new(
        cfg: SquareLatticeConfig,
        init_method: SquareLatticeInitMethod<T>,
    ) -> Result<Self, SquareLatticeConfigError> {
        let mut lattice = Self {
            cells: Tensor::<T, DenseBackend>::empty(&cfg.tensor_shape()),
            cfg,
        };

        match init_method {
            SquareLatticeInitMethod::Empty => {}
            SquareLatticeInitMethod::Uniform { val } => lattice.cells.fill(val),
            SquareLatticeInitMethod::RandomChoices {
                choices,
                weights,
                key,
            } => {
                if choices.is_empty() {
                    return Err(SquareLatticeConfigError::EmptyChoices);
                }
                let cumulative = cumulative_weights(&choices, weights.as_deref())?;
                lattice
                    .cells_mut()
                    .par_iter_mut()
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

    pub fn downsample(&self, target_shape: &[usize]) -> Self {
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
        };

        new.cells_mut()
            .par_iter_mut()
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
                let coord_old = self.boundary_index(&coord_old);
                *slot = self.cells.get(&coord_old);
            });

        new
    }

    #[inline]
    pub fn rescale(&self, target_shape: &[usize]) -> Self {
        self.downsample(target_shape)
    }
}

impl<T: Scalar> SquareLattice<T> {
    /// Returns the authoritative geometry configuration.
    #[inline]
    pub fn config(&self) -> &SquareLatticeConfig {
        &self.cfg
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        self.cells.storage().data()
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
        }
    }

    #[inline]
    pub(crate) fn boundary_index(&self, coord: &[isize]) -> Vec<isize> {
        assert_eq!(
            coord.len(),
            self.cfg.rank(),
            "lattice coordinate rank mismatch: expected {}, got {}",
            self.cfg.rank(),
            coord.len()
        );
        coord
            .iter()
            .zip(self.cfg.shape().iter())
            .map(|(&c, &axis_len)| self.cfg.boundary().normalize(c, axis_len) as isize)
            .collect()
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
        let coord = self.boundary_index(coord);
        self.cells.get_mut_for_ref(&coord)
    }

    #[inline]
    fn get_mut(&mut self, coord: &[isize]) -> &mut T {
        let coord = self.boundary_index(coord);
        self.cells.get_mut(&coord)
    }

    #[inline]
    fn set(&mut self, coord: &[isize], val: T) {
        let coord = self.boundary_index(coord);
        self.cells.set(&coord, val);
    }

    #[inline]
    fn save(&self, output_file: &PathBuf, l_target: usize) -> std::io::Result<()> {
        let target_shape = vec![l_target; self.cfg.rank()];
        crate::space::io::square_lattice::save_square_lattice(self, &target_shape, output_file)
    }

    #[inline]
    fn set_all(&mut self, val: T) {
        self.cells.fill(val);
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

trait TensorRefGet<T: Scalar> {
    fn get_mut_for_ref(&self, coord: &[isize]) -> &T;
}

impl<T: Scalar> TensorRefGet<T> for Tensor<T, DenseBackend> {
    #[inline]
    fn get_mut_for_ref(&self, coord: &[isize]) -> &T {
        let data = self.storage().data();
        let shape = self.shape();
        let mut flat = 0usize;
        for (&c, &dim) in coord.iter().zip(shape.iter()) {
            flat = flat * dim + c as usize;
        }
        &data[flat]
    }
}

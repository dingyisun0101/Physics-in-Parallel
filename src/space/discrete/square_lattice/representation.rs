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

use rand::random_range;
use rayon::prelude::*;

use crate::math::prelude::{DenseBackend, Scalar, ScalarSerde, Tensor};
use crate::space::space_trait::Space;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    Periodic,
    Reflective,
}

impl BoundaryCondition {
    #[inline]
    pub fn normalize(self, coord: isize, side_len: usize) -> usize {
        debug_assert!(side_len > 0);
        match self {
            Self::Periodic => wrap_periodic(coord, side_len),
            Self::Reflective => reflect_coordinate(coord, side_len),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SquareLatticeConfig {
    pub shape: Vec<usize>,
    pub boundary: BoundaryCondition,
}

impl SquareLatticeConfig {
    #[inline]
    pub fn new(shape: &[usize], boundary: BoundaryCondition) -> Self {
        assert!(
            !shape.is_empty(),
            "SquareLatticeConfig requires at least one axis"
        );
        assert!(
            shape.iter().all(|&n| n > 0),
            "SquareLatticeConfig requires every axis length to be nonzero; got {shape:?}"
        );
        Self {
            shape: shape.to_vec(),
            boundary,
        }
    }

    #[inline]
    pub fn periodic(shape: &[usize]) -> Self {
        Self::new(shape, BoundaryCondition::Periodic)
    }

    #[inline]
    pub fn reflective(shape: &[usize]) -> Self {
        Self::new(shape, BoundaryCondition::Reflective)
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn num_sites(&self) -> usize {
        self.shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, dim| {
                acc.checked_mul(dim)
                    .ok_or("square lattice site count overflow")
            })
            .expect("square lattice site count overflow")
    }

    #[inline]
    pub fn tensor_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[derive(Debug, Clone)]
pub enum SquareLatticeInitMethod<T: Scalar> {
    Empty,
    Uniform { val: T },
    RandomUniformChoices { choices: Vec<T> },
    SeededCenter { val: T },
}

#[derive(Debug, Clone)]
pub struct SquareLattice<T: Scalar> {
    pub cfg: SquareLatticeConfig,
    cells: Tensor<T, DenseBackend>,
}

pub trait VacancyValue: Sized + Clone {
    const VACANCY: Self;

    #[inline]
    fn vacancy() -> Self {
        Self::VACANCY
    }
}

impl VacancyValue for usize {
    const VACANCY: usize = 0;
}
impl VacancyValue for u64 {
    const VACANCY: u64 = 0;
}
impl VacancyValue for u32 {
    const VACANCY: u32 = 0;
}
impl VacancyValue for u16 {
    const VACANCY: u16 = 0;
}
impl VacancyValue for u8 {
    const VACANCY: u8 = 0;
}
impl VacancyValue for isize {
    const VACANCY: isize = 0;
}
impl VacancyValue for i64 {
    const VACANCY: i64 = 0;
}
impl VacancyValue for i32 {
    const VACANCY: i32 = 0;
}
impl VacancyValue for i16 {
    const VACANCY: i16 = 0;
}
impl VacancyValue for i8 {
    const VACANCY: i8 = 0;
}
impl VacancyValue for f64 {
    const VACANCY: f64 = 0.0;
}
impl VacancyValue for f32 {
    const VACANCY: f32 = 0.0;
}

impl<T: Scalar + VacancyValue> SquareLattice<T> {
    pub fn new(cfg: SquareLatticeConfig, init_method: SquareLatticeInitMethod<T>) -> Self {
        let mut lattice = Self {
            cells: Tensor::<T, DenseBackend>::empty(&cfg.tensor_shape()),
            cfg,
        };

        match init_method {
            SquareLatticeInitMethod::Empty => {}
            SquareLatticeInitMethod::Uniform { val } => lattice.cells.fill(val),
            SquareLatticeInitMethod::RandomUniformChoices { choices } => {
                assert!(
                    !choices.is_empty(),
                    "RandomUniformChoices requires at least one choice"
                );
                lattice
                    .cells_mut()
                    .par_iter_mut()
                    .for_each(|slot| *slot = choices[random_range(0..choices.len())]);
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

        lattice
    }

    #[inline]
    pub fn vacancy() -> T {
        T::vacancy()
    }

    #[inline]
    pub fn set_vacant(&mut self, coord: &[isize]) {
        let coord = self.boundary_index(coord);
        self.cells.set(&coord, Self::vacancy());
    }

    #[inline]
    pub fn is_vacant(&self, coord: &[isize]) -> bool {
        let coord = self.boundary_index(coord);
        self.cells.get(&coord) == Self::vacancy()
    }

    #[inline]
    pub fn fill_vacancy(&mut self) {
        self.cells.fill(Self::vacancy());
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
                .zip(self.cfg.shape.iter())
                .all(|(&new_dim, &old_dim)| new_dim <= old_dim),
            "downsample target shape must not exceed source shape: source {:?}, target {target_shape:?}",
            self.cfg.shape
        );
        if target_shape == self.cfg.shape.as_slice() {
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
        let new_cfg = SquareLatticeConfig::new(target_shape, self.cfg.boundary);
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
            .zip(self.cfg.shape.iter())
            .map(|(&c, &axis_len)| self.cfg.boundary.normalize(c, axis_len) as isize)
            .collect()
    }
}

impl<T: ScalarSerde + VacancyValue> Space<T> for SquareLattice<T> {
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

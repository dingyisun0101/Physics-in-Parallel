use std::fs::{File};
use std::io::Write;
use std::path::PathBuf;
use serde::Serialize;

use rayon::prelude::*;
use rand::random_range;

use super::super::math_fundations::scalar::Scalar;

//==========================================================================================
// ------------------------- General Formalism: Discrete Space -----------------------------
//==========================================================================================
// A general trait for all types of d-dimensional discrete spaces
// Data are stored in a flat vector, and accessed via coordinates
// Id for each site is given as a vector of usize

pub trait Space<T: Scalar> {
    fn data(&self) -> &[T];
    fn dims(&self) -> Vec<usize>;
    fn linear_size(&self) -> usize;
    fn get(&self, coord: &[usize]) -> &T;
    fn get_mut(&mut self, coord: &[usize]) -> &mut T;
    fn set(&mut self, coord: &[usize], val: T);
    fn to_adj_matrix(&mut self) -> Vec<Vec<f64>>;
    fn save(&self, output_dir: &PathBuf, l_target: usize) -> std::io::Result<()>;
    fn set_all(&mut self, val: T);
}




//==========================================================================================
// ------------------------ Space Type I: Square Lattice (Grid) ----------------------------
//==========================================================================================
// Grid is a square-lattice type d-dimensional discrete space with a fixed linear size
// It is defined by a configuration (dimension, linear size, minimum length scale, periodicity)

// --------------------------------- Basic Struct ----------------------------------
// MAX represents an empty grid
pub const VACANCY: usize = usize::MAX;

// Grid configuration data
#[derive(Debug, Clone)]
pub struct GridConfig {
    pub d: usize,           // dimension
    pub l: usize,           // maximum length scale (linear size)
    pub c: f64,             // minimum length scale (linear interval)
    pub periodic: bool,     // boundary condition
}

impl GridConfig {
    pub fn new(d: usize, l: usize, c: f64, periodic: bool ) -> Self {
        Self {
            d,
            l,
            c,
            periodic,
        }
    }
    
    #[inline]
    pub fn num_sites(&self) -> usize {
        self.l.pow(self.d as u32)
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.d, self.l]
    }
}

// Types of grid
#[derive(Debug, Clone)]
pub enum GridType<T: Scalar> {
    Empty,
    Uniform { val: T },
    RandomUniform { choices: Vec<T> },
    Dispersal { val: T },
}


// --------------------------------- Main def and impl ----------------------------------
#[derive(Debug, Clone)]
pub struct Grid<T: Scalar> {
    pub cfg: GridConfig,
    pub kind: GridType<T>,
    pub data: Vec<T>,
}

impl<T: Scalar> Grid<T> {
    pub fn new(cfg: GridConfig, kind: GridType<T>) -> Self {
        let mut data = vec![T::default(); cfg.num_sites()];

        match &kind {
            GridType::Empty => {}
            GridType::Uniform { val } => {
                data.par_iter_mut()
                    .for_each(|slot| *slot = val.clone());
            }
            GridType::RandomUniform { choices } => {
                assert!(!choices.is_empty());
                data.par_iter_mut()
                    .for_each(|slot| {
                        let i = random_range(0..choices.len());
                        *slot = choices[i].clone();
                    });
            }
            GridType::Dispersal { val } => {
                let mut idx = 0usize;
                for _ in 0..cfg.d {
                    idx = idx * cfg.l + cfg.l / 2;
                }
                data[idx] = val.clone();
            }
        }

        Self { cfg, kind, data }
    }

    #[inline(always)]
    fn coord_to_index(&self, coord: &[usize]) -> usize {
        debug_assert_eq!(coord.len(), self.cfg.d, "coord len {} != d {}", coord.len(), self.cfg.d);
        let l = self.cfg.l;
        let periodic = self.cfg.periodic;

        let mut flat = 0usize;
        for &c in coord {
            let cc = if periodic { c % l } else {
                debug_assert!(c < l, "coord component {} >= l {}", c, l);
                c
            };
            flat = flat * l + cc; // MSB-first: [x,y] -> x*l + y
        }
        flat
    }

    #[inline(always)]
    pub fn shape(&self) -> [usize; 2] {
        [self.cfg.d, self.cfg.l]
    }

    pub fn rescale(&self, l_new: usize) -> Self {
        if l_new >= self.cfg.l {
            return self.clone();
        }

        let d = self.cfg.d;
        let scale = self.cfg.l as f64 / l_new as f64;

        let new_cfg = GridConfig {
            d,
            l: l_new,
            c: self.cfg.c,
            periodic: self.cfg.periodic,
        };

        let mut new = Grid {
            cfg: new_cfg.clone(),
            kind: self.kind.clone(),
            data: vec![T::default(); new_cfg.num_sites()],
        };

        new.data
            .par_iter_mut()
            .enumerate()
            .for_each(|(flat, slot)| {
                let mut rem = flat;
                let mut coord_new = vec![0usize; d];
                for k in (0..d).rev() {
                    coord_new[k] = rem % l_new;
                    rem /= l_new;
                }

                let coord_old: Vec<usize> = coord_new
                    .iter()
                    .map(|&x| (x as f64 * scale).floor() as usize)
                    .map(|x| x.min(self.cfg.l - 1))
                    .collect();

                *slot = self.get(&coord_old).clone();
            });

        new
    }
}

// --------------------------------- Trait impl ----------------------------------
impl<T: Scalar> Space<T> for Grid<T> {
    fn data(&self) -> &[T] {
        &self.data
    }

    fn dims(&self) -> Vec<usize> {
        vec![self.cfg.d, self.cfg.l]
    }

    fn linear_size(&self) -> usize {
        self.data.len()
    }

    fn get(&self, coord: &[usize]) -> &T {
        let i = self.coord_to_index(coord);
        &self.data[i]
    }

    fn get_mut(&mut self, coord: &[usize]) -> &mut T {
        let i = self.coord_to_index(coord);
        &mut self.data[i]
    }

    fn set(&mut self, coord: &[usize], val: T) {
        let i = self.coord_to_index(coord);
        self.data[i] = val;
    }

    fn set_all(&mut self, val: T) {
        self.data
            .par_iter_mut()
            .for_each(|x| *x = val.clone());
    }

    fn to_adj_matrix(&mut self) -> Vec<Vec<f64>> {
        let d = self.cfg.d;
        let l = self.cfg.l;
        let n = self.linear_size();
        let periodic = self.cfg.periodic;

        let mut adj = vec![vec![0.0_f64; n]; n];

        let decode = |mut idx: usize| {
            let mut coord = vec![0usize; d];
            for k in (0..d).rev() {
                coord[k] = idx % l;
                idx /= l;
            }
            coord
        };

        for i in 0..n {
            let coord = decode(i);

            for dim in 0..d {
                for &delta in &[-1isize, 1isize] {
                    let mut neigh = coord.clone();
                    let x = neigh[dim] as isize + delta;
                    if periodic {
                        let m = l as isize;
                        neigh[dim] = ((x % m) + m) as usize % l;
                    } else if x < 0 || x >= l as isize {
                        continue;
                    } else {
                        neigh[dim] = x as usize;
                    }

                    let mut lin = 0usize;
                    for &c in &neigh {
                        lin = lin * l + c;
                    }
                    adj[i][lin] = 1.0;
                }
            }
        }

        adj
    }

    fn save(&self, output_dir: &PathBuf, l_target: usize) -> std::io::Result<()> {
        save_grid(self, l_target, output_dir)
    }
}


pub fn save_grid<T>(grid: &Grid<T>, l_target: usize, output_dir: &PathBuf) -> std::io::Result<()>
where
    T: Scalar + Serialize,
{
    let grid_to_save = grid.rescale(l_target);

    #[derive(Serialize)]
    struct GridJson<'a, T> {
        shape: [usize; 2],
        data: &'a [T],
    }

    let json_data = GridJson {
        shape: grid_to_save.shape(),
        data: &grid_to_save.data,
    };

    let json = serde_json::to_string_pretty(&json_data)
        .expect("Failed to serialize grid");

    let mut file = File::create(output_dir)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}












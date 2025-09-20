// src/sim/discrete/grid.rs
/*!
A **discrete lattice/grid** for d-dimensional simulations with:

- **Type-aware vacancies** via a `VacancyValue` trait (sentinel values for “empty” cells).
- **Ergonomic coordinate access** with `isize` indices and **periodic wrap** or **hard clamp**.
- **Parallel initialization and mutation** using `rayon`.
- **Configurable generators** (`Empty`, `Uniform`, `RandomUniform`, `Dispersal`).
- **Adjacency matrix** construction for nearest-neighbor graphs (±1 step per axis).
- **Downsampled saving**: optional integer down-scaling before writing JSON.

This module mirrors the style of the dense tensor implementation: clear invariants,
aggressive inlining, and rich documentation.

# Highlights

- `VacancyValue` defines a sentinel for “vacant” cells per scalar type.
- `GridConfig { d, l, c, periodic }`:
  - `d`: rank (number of spatial axes),
  - `l`: linear side length per axis (total sites = `l^d`),
  - `c`: arbitrary physical scale/unit (kept meta),
  - `periodic`: wrap vs clamp.
- `GridInitMethod<T>` for one-shot initialization of grid states.
- `Grid<T>`:
  - data stored row-major (axis 0 … axis d-1; last axis varies fastest),
  - `coord_to_index([i0, …, i_{d-1}]) = (((i0 * l + i1) * l + …) * l + i_{d-1})`.
- `Space<T>` trait: read/write cells, fill all, build adjacency, save (with optional downscale).

> **Note**  
> This file assumes a project-wide `Scalar` trait and `serde`/`rayon` dependencies.
> Random choice uses `rand::random_range` (available in `rand` ≥ 0.9).  
> If you’re on `rand` 0.8, replace with `thread_rng().gen_range(0..choices.len())`.

*/

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use serde::Serialize;
use rayon::prelude::*;
use rand::random_range;

use crate::math::scalar::Scalar;
use crate::space::space_trait::Space;


// ======================================================================================
// --------------------------- Vacancy Sentinel (type-aware) -----------------------------
// ======================================================================================

/**
`VacancyValue` provides a **type-specific sentinel** representing a vacant/empty cell.

- Unsigned/signed integers: `*::MAX`
- Floats: `f32::MAX` / `f64::MAX` (swap to `NaN` if that better fits your pipeline)

The exact sentinel is **your contract**: downstream code should treat `VacancyValue::vacancy()`
as “no occupant / no data.”
*/
pub trait VacancyValue: Sized + Clone {
    /// Type-specific sentinel for "vacant".
    const VACANCY: Self;

    // Optional ergonomic non-const helper:
    #[inline]
    fn vacancy() -> Self { Self::VACANCY }
}

// Unsigned integers
impl VacancyValue for usize { const VACANCY: usize = usize::MAX; }
impl VacancyValue for u64  { const VACANCY: u64  = u64::MAX; }
impl VacancyValue for u32  { const VACANCY: u32  = u32::MAX; }
impl VacancyValue for u16  { const VACANCY: u16  = u16::MAX; }
impl VacancyValue for u8   { const VACANCY: u8   = u8::MAX;  }

// Signed integers
impl VacancyValue for isize { const VACANCY: isize = isize::MAX; }
impl VacancyValue for i64   { const VACANCY: i64   = i64::MAX; }
impl VacancyValue for i32   { const VACANCY: i32   = i32::MAX; }
impl VacancyValue for i16   { const VACANCY: i16   = i16::MAX; }
impl VacancyValue for i8    { const VACANCY: i8    = i8::MAX;  }

// Floats (keep as MAX; switch to NAN/INFINITY if your math prefers)
impl VacancyValue for f64 { const VACANCY: f64 = f64::MAX; }
impl VacancyValue for f32 { const VACANCY: f32 = f32::MAX; }


// ======================================================================================
// --------------------------------- GridConfig -----------------------------------------
// ======================================================================================

/**
Configuration for a **d-dimensional cubic lattice** with equal side length per axis.

- `d`: rank (e.g., 1D line, 2D square, 3D cube, …)
- `l`: side length per axis (total sites = `l^d`)
- `c`: arbitrary **scale/constant** (kept as meta for physics)
- `periodic`: if `true`, out-of-bounds indices **wrap**; else they **clamp** to `[0, l-1]`
*/
#[derive(Debug, Clone, Serialize)]
pub struct GridConfig {
    pub d: usize,
    pub l: usize,
    pub c: f64,
    pub periodic: bool,
}

impl GridConfig {
    /// Create a new config. Panics if `d == 0` or `l == 0`.
    #[inline]
    pub fn new(d: usize, l: usize, c: f64, periodic: bool) -> Self {
        assert!(d > 0 && l > 0, "GridConfig requires d>0 and l>0; got d={d}, l={l}");
        Self { d, l, c, periodic }
    }

    /// Total site count `l^d`.
    #[inline]
    pub fn num_sites(&self) -> usize {
        self.l.pow(self.d as u32)
    }

    /// Return `[d, l]`. (Metadata shape; **not** the flattened length.)
    #[inline]
    pub fn shape(&self) -> [usize; 2] { [self.d, self.l] }
}



// ======================================================================================
// -------------------------------- GridInitMethod --------------------------------------
// ======================================================================================

/**
One-shot **initialization strategy** for a grid.

- `Empty`: leave `data` as default values (or set to vacancy in your pipeline).
- `Uniform { val }`: fill the grid with `val`.
- `RandomUniformChoices { choices }`: pick uniformly among `choices` for each site.
- `Dispersal { val }`: set a **single central** site to `val` (others remain default).
*/
#[derive(Debug, Clone, Serialize)]
pub enum GridInitMethod<T: Scalar> {
    Empty,
    Uniform { val: T },
    RandomUniformChoices { choices: Vec<T> },
    SeededCenter { val: T },
}



// ======================================================================================
// ------------------------------------ Grid --------------------------------------------
// ======================================================================================

/**
`Grid<T>`: a row-major, d-dimensional **lattice** with `l^d` sites.

- Storage: `Vec<T>` of length `cfg.num_sites()`.
- Indexing order: **row-major** (last axis fastest).
- Uses `VacancyValue` to standardize “empty” cells across scalar types.

# Invariants
- `data.len() == cfg.num_sites()`.
- `cfg.d > 0`, `cfg.l > 0` (enforced by `GridConfig::new`).
*/
#[derive(Debug, Clone, Serialize)]
pub struct Grid<T: Scalar> {
    /// Configuration (rank, side length, scale, periodicity).
    pub cfg: GridConfig,
    /// Row-major storage for all sites (length = `l^d`).
    pub data: Vec<T>,
}



// ======================================================================================
// --------------------- Vacancy operations (type-aware) --------------------------------
// ======================================================================================

impl<T: Scalar + VacancyValue> Grid<T> {
    /// The **sentinel** for vacancy for `T`.
    #[inline]
    pub fn vacancy() -> T { T::vacancy() }

    /// Mark a single coordinate as **vacant**.
    ///
    /// Negative coordinates are **wrapped** if periodic, or **clamped** otherwise.
    #[inline]
    pub fn set_vacant(&mut self, coord: &[isize]) {
        let i = self.coord_to_index(coord);
        self.data[i] = Self::vacancy();
    }

    /// Check whether the cell at `coord` is **vacant** (equal to the sentinel).
    ///
    /// > If your `VacancyValue` is a floating sentinel (e.g., `f64::MAX`),
    /// > this uses **exact equality** to the sentinel.
    #[inline]
    pub fn is_vacant(&self, coord: &[isize]) -> bool {
        let i = self.coord_to_index(coord);
        self.data[i] == Self::vacancy()
    }

    /// Fill the entire grid with the **vacancy** sentinel (parallel).
    #[inline]
    pub fn fill_vacancy(&mut self) {
        let v = Self::vacancy();
        self.data.par_iter_mut().for_each(|x| *x = v.clone());
    }
}




// ======================================================================================
// ------------------------ Constructors & core helpers ---------------------------------
// ======================================================================================

impl<T: Scalar + VacancyValue> Grid<T> {
    /**
    Build a new grid with the given `cfg` and initialization `init_method`.

    - `Empty`: leaves `data` at `T::default()`.  
      If you want **Empty → vacancy**, call `fill_vacancy()` after construction.
    - `Uniform { val }`: `val` everywhere (parallel).
    - `RandomUniform { choices }`: uniform random pick per site.
      - **Panics** if `choices` is empty.
    - `Dispersal { val }`: set the **center** site to `val` (others remain default).
    */
    #[inline]
    pub fn new(cfg: GridConfig, init_method: GridInitMethod<T>) -> Self {
        let mut data = vec![T::default(); cfg.num_sites()];

        match &init_method {
            GridInitMethod::Empty => {
                // If you prefer Empty→Vacancy, uncomment:
                // let v = T::vacancy();
                // data.par_iter_mut().for_each(|x| *x = v.clone());
            }
            GridInitMethod::Uniform { val } => {
                data.par_iter_mut().for_each(|slot| *slot = val.clone());
            }
            GridInitMethod::RandomUniformChoices { choices } => {
                assert!(!choices.is_empty(), "RandomUniform requires non-empty `choices`");
                data.par_iter_mut().for_each(|slot| {
                    let i = random_range(0..choices.len());
                    *slot = choices[i].clone();
                });
            }
            GridInitMethod::SeededCenter { val } => {
                // Put a single non-vacant marker at the **center**.
                // Center coordinate is (l/2, l/2, ..., l/2).
                let mut idx = 0usize;
                for _ in 0..cfg.d { idx = idx * cfg.l + cfg.l / 2; }
                data[idx] = val.clone();
                // If desired, set all others to vacancy:
                // let v = T::vacancy();
                // data.iter_mut().enumerate().for_each(|(k, x)| if k != idx { *x = v.clone(); });
            }
        }

        Self { cfg, data }
    }

    // --- Private helpers -----------------------------------------------------

    /// Return a compact metadata “shape” as `[d, l]`.
    #[inline(always)]
    fn shape(&self) -> [usize; 2] { [self.cfg.d, self.cfg.l] }

    /// **Wrap** (periodic) or **clamp** (non-periodic) a single coordinate into `[0, l-1]`.
    #[inline(always)]
    fn wrap_or_clamp(&self, c: isize) -> usize {
        let l = self.cfg.l as isize;
        if self.cfg.periodic {
            // Proper mathematical modulo for negatives: ((c % l) + l) % l
            (((c % l) + l) % l) as usize
        } else {
            c.clamp(0, l - 1) as usize
        }
    }

    /// Convert a multi-index to a **row-major** flat index.
    ///
    /// # Panics
    /// - If rank mismatch.
    #[inline(always)]
    fn coord_to_index(&self, coord: &[isize]) -> usize {
        debug_assert_eq!(coord.len(), self.cfg.d, "rank mismatch: coord={:?}, d={}", coord, self.cfg.d);
        let l = self.cfg.l;
        let mut flat = 0usize;
        for &c in coord {
            let cc = self.wrap_or_clamp(c);
            flat = flat * l + cc;
        }
        flat
    }
}




// ======================================================================================
// -------------------------- Space impl (core ops) -------------------------------------
// ======================================================================================

impl<T: Scalar + VacancyValue + Serialize> Space<T> for Grid<T> {
    /// Borrow backing data.
    #[inline] fn data(&self) -> &[T] { &self.data }

    /// Return metadata dims `[d, l]` (total sites is `l^d`).
    #[inline] fn dims(&self) -> Vec<usize> { vec![self.cfg.d, self.cfg.l] }

    /// Total site count `l^d`.
    #[inline] fn linear_size(&self) -> usize { self.data.len() }

    /// Safe read at multi-index `coord`.
    #[inline]
    fn get(&self, coord: &[isize]) -> &T {
        let i = self.coord_to_index(coord);
        &self.data[i]
    }

    /// Safe mutable read at multi-index `coord`.
    #[inline]
    fn get_mut(&mut self, coord: &[isize]) -> &mut T {
        let i = self.coord_to_index(coord);
        &mut self.data[i]
    }

    /// Safe write at multi-index `coord`.
    #[inline]
    fn set(&mut self, coord: &[isize], val: T) {
        let i = self.coord_to_index(coord);
        self.data[i] = val;
    }

    /// Parallel fill of all sites with `val`.
    #[inline]
    fn set_all(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|x| *x = val.clone());
    }

    /**
    Build a **nearest-neighbor adjacency matrix** for the grid.

    - Graph has `n = l^d` nodes.
    - An undirected edge exists between sites that differ by `±1` along **exactly one** axis.
    - Periodic vs clamped boundary matches `cfg.periodic`.

    Returns a dense `n×n` matrix of `0.0/1.0`.  
    For large grids this is memory-heavy; prefer sparse representations in production.
    */
    fn to_adj_matrix(&mut self) -> Vec<Vec<f64>> {
        let d = self.cfg.d;
        let l = self.cfg.l as isize;
        let n = self.linear_size();

        let mut adj = vec![vec![0.0_f64; n]; n];

        // Decode flat index → coordinates (row-major, last axis fastest).
        let decode = |mut idx: usize| {
            let mut coord = vec![0isize; d];
            for k in (0..d).rev() {
                coord[k] = (idx % self.cfg.l) as isize;
                idx /= self.cfg.l;
            }
            coord
        };

        for i in 0..n {
            let coord = decode(i);

            for dim in 0..d {
                for &delta in &[-1isize, 1isize] {
                    // neighbor by ±1 step on axis `dim`
                    let mut neigh = coord.clone();
                    let x = neigh[dim] + delta;
                    neigh[dim] = if self.cfg.periodic {
                        ((x % l) + l) % l
                    } else {
                        x.clamp(0, l - 1)
                    };

                    // Encode neighbor coords → flat index
                    let mut lin = 0usize;
                    for &c in &neigh {
                        lin = lin * self.cfg.l + (c as usize);
                    }
                    adj[i][lin] = 1.0;
                }
            }
        }
        adj
    }

    /// Save the grid to `output_file` in JSON after optional **downscaling** to side `l_target`.
    ///
    /// The JSON schema is:
    /// ```json
    /// { "shape": [d, l_target], "data": [ ... length = l_target^d ... ] }
    /// ```
    fn save(&self, output_file: &PathBuf, l_target: usize) -> std::io::Result<()> {
        save_grid(self, l_target, output_file)
    }
}



// ======================================================================================
// ------------------------------ Downscale & Save --------------------------------------
// ======================================================================================

impl<T: Scalar + VacancyValue> Grid<T> {
    /**
    Create a **downsampled** copy with side length `l_new`.

    - If `l_new >= l`, this returns a **clone** (no upsampling).
    - If `l_new < l`, each new coordinate `(i_new)` maps to  
      `i_old = floor(i_new * (l / l_new))` **along each axis** (nearest-lower pick).

    This is **not an average**; it’s a point-sample mapping suitable for
    categorical grids (e.g., **occupancy** / **cell types**).
    */
    #[inline]
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
            data: vec![T::default(); new_cfg.num_sites()],
        };

        // Parallel: map each new flat index → new coords → old coords → copy value.
        new.data
            .par_iter_mut()
            .enumerate()
            .for_each(|(flat, slot)| {
                // Decode flat index in the new grid into coordinates
                let mut rem = flat;
                let mut coord_new = vec![0usize; d];
                for k in (0..d).rev() {
                    coord_new[k] = rem % l_new;
                    rem /= l_new;
                }

                // Scale back into old coordinate system (nearest-lower)
                let coord_old: Vec<isize> = coord_new
                    .iter()
                    .map(|&x| (x as f64 * scale).floor() as isize)
                    .collect();

                *slot = self.get(&coord_old).clone();
            });

        new
    }
}

/// Save a grid to a JSON file after optional **downscaling** to side `l_target`.
///
/// - `grid`: source grid
/// - `l_target`: side length of the saved grid (use same value as `grid.cfg.l` to avoid scaling)
/// - `output_file`: **file path** to write to (not a directory)
pub fn save_grid<T>(grid: &Grid<T>, l_target: usize, output_file: &PathBuf) -> std::io::Result<()>
where
    T: Scalar + Serialize + VacancyValue,
{
    let grid_to_save = grid.rescale(l_target);

    #[derive(Serialize)]
    struct GridJson<'a, T> {
        /// Metadata “shape”: `[d, l]` (note: total data length is `l^d`)
        shape: [usize; 2],
        /// Row-major flattened data of length `l^d`
        data: &'a [T],
    }

    let json_data = GridJson {
        shape: grid_to_save.shape(),
        data: &grid_to_save.data,
    };

    let json = serde_json::to_string_pretty(&json_data)
        .expect("Failed to serialize grid to JSON");

    let mut file = File::create(output_file)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

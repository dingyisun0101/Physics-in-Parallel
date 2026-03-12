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

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use rayon::prelude::*;
use rand::random_range;
use ndarray::{ArrayD, IxDyn};

use crate::io::json::{FlatPayload, FlatPayloadRef, FromJsonPayload, ToJsonPayload};
use crate::math::prelude::{Scalar, ScalarSerde};
use crate::space::space_trait::Space;


// ======================================================================================
// --------------------------- Vacancy Sentinel (type-aware) -----------------------------
// ======================================================================================

/**
`VacancyValue` provides a **type-specific sentinel** representing a vacant/empty cell.

- Unsigned/signed integers: `0`
- Floats: `0.0`

The exact sentinel is **your contract**: downstream code should treat `VacancyValue::vacancy()`
as “no occupant / no data.”
*/
pub trait VacancyValue: Sized + Clone {
    /// Type-specific sentinel for "vacant".
    const VACANCY: Self;

    // Optional ergonomic non-const helper:
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `vacancy` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    ///   - (none): This function takes no explicit parameters.
    fn vacancy() -> Self { Self::VACANCY }
}

// Unsigned integers
impl VacancyValue for usize { const VACANCY: usize = 0; }
impl VacancyValue for u64  { const VACANCY: u64  = 0; }
impl VacancyValue for u32  { const VACANCY: u32  = 0; }
impl VacancyValue for u16  { const VACANCY: u16  = 0; }
impl VacancyValue for u8   { const VACANCY: u8   = 0; }

// Signed integers
impl VacancyValue for isize { const VACANCY: isize = 0; }
impl VacancyValue for i64   { const VACANCY: i64   = 0; }
impl VacancyValue for i32   { const VACANCY: i32   = 0; }
impl VacancyValue for i16   { const VACANCY: i16   = 0; }
impl VacancyValue for i8    { const VACANCY: i8    = 0; }

// Floats
impl VacancyValue for f64 { const VACANCY: f64 = 0.0; }
impl VacancyValue for f32 { const VACANCY: f32 = 0.0; }


// ======================================================================================
// --------------------------------- GridConfig -----------------------------------------
// ======================================================================================

/**
Configuration for a **d-dimensional cubic lattice** with equal side length per axis.

- `d`: rank (e.g., 1D line, 2D square, 3D cube, …)
- `l`: side length per axis (total sites = `l^d`)
- `periodic`: if `true`, out-of-bounds indices **wrap**; else they **clamp** to `[0, l-1]`
*/
#[derive(Debug, Clone, Serialize)]
pub struct GridConfig {
    pub d: usize,
    pub l: usize,
    pub periodic: bool,
}

impl GridConfig {
    /// Create a new config. Panics if `d == 0` or `l == 0`.
    #[inline]
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `d` (`usize`): Parameter of type `usize` used by `new`.
    ///   - `l` (`usize`): Parameter of type `usize` used by `new`.
    ///   - `periodic` (`bool`): Parameter of type `bool` used by `new`.
    pub fn new(d: usize, l: usize, periodic: bool) -> Self {
        assert!(d > 0 && l > 0, "GridConfig requires d>0 and l>0; got d={d}, l={l}");
        Self { d, l, periodic }
    }

    /// Total site count `l^d`.
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `num_sites` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn num_sites(&self) -> usize {
        self.l.pow(self.d as u32)
    }

    /// Return `[d, l]`. (Metadata shape; **not** the flattened length.)
    #[inline]
    /// Annotation:
    /// - Purpose: Returns the logical shape metadata.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
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
#[derive(Debug, Clone)]
pub struct Grid<T: Scalar> {
    /// Configuration (rank, side length, scale, periodicity).
    pub cfg: GridConfig,
    /// Row-major storage for all sites (length = `l^d`).
    pub data: Vec<T>,
}

impl<T> Serialize for Grid<T>
where
    T: Scalar + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_json_payload()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for Grid<T>
where
    T: Scalar + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = FlatPayload::<T>::deserialize(deserializer)?;
        <Self as FromJsonPayload>::from_json_payload(payload).map_err(serde::de::Error::custom)
    }
}

impl<T: Scalar> Grid<T> {
    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `ndarry` input.
    /// - Parameters:
    ///   - `array` (`&ArrayD<T>`): ndarray input used for conversion/interoperability.
    ///   - `periodic` (`bool`): Parameter of type `bool` used by `from_ndarry`.
    pub fn from_ndarry(array: &ArrayD<T>, periodic: bool) -> Self {
        let owned = array.to_owned();
        let shape = owned.shape().to_vec();
        assert!(!shape.is_empty(), "Grid::from_ndarry: shape must be non-empty");
        let l = shape[0];
        assert!(
            shape.iter().all(|&dim| dim == l),
            "Grid::from_ndarry: expected cubic shape, got {shape:?}"
        );
        let cfg = GridConfig::new(shape.len(), l, periodic);
        let (data, _) = owned.into_raw_vec_and_offset();
        Self {
            cfg,
            data,
        }
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_ndarray(&self) -> ArrayD<T> {
        let shape = vec![self.cfg.l; self.cfg.d];
        ArrayD::from_shape_vec(IxDyn(&shape), self.data.clone())
            .expect("Grid::to_ndarray: shape/data length mismatch")
    }

    #[inline(always)]
    fn ndarray_shape(&self) -> Vec<usize> {
        vec![self.cfg.l; self.cfg.d]
    }

    #[inline(always)]
    fn serde_kind(&self) -> &'static str {
        if self.cfg.periodic {
            "grid_periodic"
        } else {
            "grid_clamped"
        }
    }
}

#[inline(always)]
fn parse_grid_kind(kind: &str) -> Result<bool, String> {
    match kind {
        "grid_periodic" | "grid" => Ok(true),
        "grid_clamped" => Ok(false),
        _ => Err(format!(
            "grid kind must be 'grid_periodic' or 'grid_clamped'; got '{kind}'"
        )),
    }
}



// ======================================================================================
// --------------------- Vacancy operations (type-aware) --------------------------------
// ======================================================================================

impl<T: Scalar + VacancyValue> Grid<T> {
    /// The **sentinel** for vacancy for `T`.
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `vacancy` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    ///   - (none): This function takes no explicit parameters.
    pub fn vacancy() -> T { T::vacancy() }

    /// Mark a single coordinate as **vacant**.
    ///
    /// Negative coordinates are **wrapped** if periodic, or **clamped** otherwise.
    #[inline]
    /// Annotation:
    /// - Purpose: Sets the `vacant` value.
    /// - Parameters:
    ///   - `coord` (`&[isize]`): Coordinate input used for spatial addressing.
    pub fn set_vacant(&mut self, coord: &[isize]) {
        let i = self.coord_to_index(coord);
        self.data[i] = Self::vacancy();
    }

    /// Check whether the cell at `coord` is **vacant** (equal to the sentinel).
    ///
    /// > If your `VacancyValue` is a floating sentinel (e.g., `0.0`),
    /// > this uses **exact equality** to the sentinel.
    #[inline]
    /// Annotation:
    /// - Purpose: Checks whether `vacant` condition is true.
    /// - Parameters:
    ///   - `coord` (`&[isize]`): Coordinate input used for spatial addressing.
    pub fn is_vacant(&self, coord: &[isize]) -> bool {
        let i = self.coord_to_index(coord);
        self.data[i] == Self::vacancy()
    }

    /// Fill the entire grid with the **vacancy** sentinel (parallel).
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `fill_vacancy` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
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
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `cfg` (`GridConfig`): Parameter of type `GridConfig` used by `new`.
    ///   - `init_method` (`GridInitMethod<T>`): Parameter of type `GridInitMethod<T>` used by `new`.
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

    /// **Wrap** (periodic) or **clamp** (non-periodic) a single coordinate into `[0, l-1]`.
    #[inline(always)]
    /// Annotation:
    /// - Purpose: Executes `wrap_or_clamp` logic for this module.
    /// - Parameters:
    ///   - `c` (`isize`): Parameter of type `isize` used by `wrap_or_clamp`.
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
    /// Annotation:
    /// - Purpose: Executes `coord_to_index` logic for this module.
    /// - Parameters:
    ///   - `coord` (`&[isize]`): Coordinate input used for spatial addressing.
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

impl<T: ScalarSerde + VacancyValue> Space<T> for Grid<T> {
    /// Borrow backing data.
    /// Annotation:
    /// - Purpose: Executes `data` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn data(&self) -> &[T] { &self.data }

    /// Return metadata dims `[d, l]` (total sites is `l^d`).
    /// Annotation:
    /// - Purpose: Executes `dims` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn dims(&self) -> Vec<usize> { vec![self.cfg.d, self.cfg.l] }

    /// Total site count `l^d`.
    /// Annotation:
    /// - Purpose: Executes `linear_size` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    #[inline] fn linear_size(&self) -> usize { self.data.len() }

    /// Safe read at multi-index `coord`.
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `get` logic for this module.
    /// - Parameters:
    ///   - `coord` (`&[isize]`): Coordinate input used for spatial addressing.
    fn get(&self, coord: &[isize]) -> &T {
        let i = self.coord_to_index(coord);
        &self.data[i]
    }

    /// Safe mutable read at multi-index `coord`.
    #[inline]
    /// Annotation:
    /// - Purpose: Returns the `mut` value.
    /// - Parameters:
    ///   - `coord` (`&[isize]`): Coordinate input used for spatial addressing.
    fn get_mut(&mut self, coord: &[isize]) -> &mut T {
        let i = self.coord_to_index(coord);
        &mut self.data[i]
    }

    /// Safe write at multi-index `coord`.
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `set` logic for this module.
    /// - Parameters:
    ///   - `coord` (`&[isize]`): Coordinate input used for spatial addressing.
    ///   - `val` (`T`): Value provided by caller for write/update behavior.
    fn set(&mut self, coord: &[isize], val: T) {
        let i = self.coord_to_index(coord);
        self.data[i] = val;
    }

    /// Parallel fill of all sites with `val`.
    #[inline]
    /// Annotation:
    /// - Purpose: Sets the `all` value.
    /// - Parameters:
    ///   - `val` (`T`): Value provided by caller for write/update behavior.
    fn set_all(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|x| *x = val.clone());
    }

    /// Save the grid to `output_file` in JSON after optional **downscaling** to side `l_target`.
    ///
    /// The JSON schema is:
    /// ```json
    /// {
    ///   "kind": "grid_periodic" | "grid_clamped",
    ///   "shape": [l_target, l_target, ...],
    ///   "data": [ ... length = l_target^d ... ]
    /// }
    /// ```
    /// Annotation:
    /// - Purpose: Executes `save` logic for this module.
    /// - Parameters:
    ///   - `output_file` (`&PathBuf`): Parameter of type `&PathBuf` used by `save`.
    ///   - `l_target` (`usize`): Parameter of type `usize` used by `save`.
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
    /// Annotation:
    /// - Purpose: Executes `rescale` logic for this module.
    /// - Parameters:
    ///   - `l_new` (`usize`): Parameter of type `usize` used by `rescale`.
    pub fn rescale(&self, l_new: usize) -> Self {
        if l_new >= self.cfg.l {
            return self.clone();
        }

        let d = self.cfg.d;
        let scale = self.cfg.l as f64 / l_new as f64;

        let new_cfg = GridConfig {
            d,
            l: l_new,
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

                let i_old = self.coord_to_index(&coord_old);
                *slot = self.data[i_old].clone();
            });

        new
    }
}

impl<T: ScalarSerde + VacancyValue> Grid<T> {
    #[inline]
    /// - Purpose: Serializes this grid into pretty JSON text.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.to_json_string()
    }
}

impl<T> ToJsonPayload for Grid<T>
where
    T: Scalar + Serialize,
{
    type Payload = FlatPayload<T>;

    fn to_json_payload(&self) -> Result<Self::Payload, serde_json::Error> {
        Ok(FlatPayload::new(
            self.serde_kind(),
            self.ndarray_shape(),
            self.data.clone(),
        ))
    }
}

impl<T> FromJsonPayload for Grid<T>
where
    T: Scalar + DeserializeOwned,
{
    type Payload = FlatPayload<T>;

    fn from_json_payload(payload: Self::Payload) -> Result<Self, String> {
        let periodic = parse_grid_kind(&payload.kind)?;
        let expected_len = payload.validate_shape("grid")?;
        if payload.data.len() != expected_len {
            return Err(format!(
                "grid data length mismatch: expected {expected_len}, got {}",
                payload.data.len()
            ));
        }

        let l = payload.shape[0];
        if payload.shape.iter().any(|&dim| dim != l) {
            return Err(format!(
                "grid shape must be cubic (all dimensions equal), got {:?}",
                payload.shape
            ));
        }

        Ok(Self {
            cfg: GridConfig::new(payload.shape.len(), l, periodic),
            data: payload.data,
        })
    }
}

/// Save a grid to a JSON file after optional **downscaling** to side `l_target`.
///
/// - `grid`: source grid
/// - `l_target`: side length of the saved grid (use same value as `grid.cfg.l` to avoid scaling)
/// - `output_file`: **file path** to write to (not a directory)
pub fn save_grid<T>(
    grid: &Grid<T>,
    l_target: usize,
    output_file: &PathBuf,
) -> std::io::Result<()>
where
    T: ScalarSerde + VacancyValue,
{
    let grid_to_save = grid.rescale(l_target);
    let shape = grid_to_save.ndarray_shape();

    let json_data = FlatPayloadRef {
        kind: grid_to_save.serde_kind(),
        shape: &shape,
        data: &grid_to_save.data,
    };

    let json = serde_json::to_string_pretty(&json_data)
        .expect("Failed to serialize grid to JSON");

    let mut file = File::create(output_file)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

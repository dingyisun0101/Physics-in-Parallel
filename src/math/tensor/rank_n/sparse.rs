// src/math_foundations/tensor/sparse.rs
/*!
A **hash-backed sparse N-D tensor** where only nonzeros are stored.

- **Storage:** `AHashMap<flat_index, T>`; zeros are implicit (not stored).
- **Layout:** row-major linearization for flat indices (same as `dense::Tensor`).
- **Scalars:** `T` implements your project’s `Scalar` trait (reals/complex/etc.).
- **Elementwise ops:** operate on the **union** of nonzero indices; results that
  become zero are dropped (sparseness preserved).
- **Parallelism:** `rayon` used for binary ops and many transforms; when
  `AHashMap` must be consumed, we use `.into_iter().par_bridge()`.
- **Computation-only scope:** JSON, ndarray, and string interop live under
  `math::io`.

This module mirrors the dense tensor API where it makes sense, and defers to
`dense::Tensor` for convenient interop via `to_dense()` / `from_dense()`.

# Update — Access Semantics (Important!)

Multi-index accessors (`index`, `get`, `set`, and trait-level `get_mut`) accept `&[isize]`
and apply **toroidal (periodic) wrapping** on each axis:

- Axis index `a` maps to `((a % dim) + dim) % dim` (Euclidean modulo).
- Negative indices are allowed (`-1` = last, `-2` = second last, ...).
- **No out-of-bounds panics** from indexing (rank mismatch remains a debug assert).

Every multi-index accessor deterministically targets a valid location; implicit zeros remain zero unless set.

*/

use ahash::AHashMap;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;
use std::ops::{Add, BitAnd, Div, Mul, Sub};

use super::dense::{Tensor as TensorDense, checked_num_elements};
use super::tensor_trait::TensorTrait;
use crate::math::scalar::{Scalar, ScalarCastError};

// ===================================================================
// --------------------------- Struct Def ----------------------------
// ===================================================================

/// Sparse N-D tensor: only nonzero entries are kept in a hash map.
///
/// - `shape`: dimension sizes, rank = `shape.len()`.
/// - `data`: map from row-major flat index → value `T` (nonzero only).
///
/// # Invariants
/// - `shape.len() >= 1`.
/// - `data` contains no zeros (`T::zero()` is pruned on insert/ops).
#[derive(Clone, Debug)]
pub struct Tensor<T: Scalar> {
    shape: Vec<usize>,
    data: AHashMap<usize, T>, // flat index -> value (non-zero)
}

// ===================================================================
// ------------------------- Size & Helpers --------------------------
// ===================================================================

impl<T: Scalar> Tensor<T> {
    /// Total number of sites (dense size) = product of dimensions.
    #[inline(always)]
    /// Details:
    /// - Purpose: Returns the current length/size.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn len_dense(&self) -> usize {
        checked_num_elements(&self.shape, "sparse tensor")
    }

    /// Rank (number of dimensions).
    #[inline(always)]
    /// Details:
    /// - Purpose: Executes `rank` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Shape slice.
    #[inline(always)]
    /// Details:
    /// - Purpose: Returns the logical shape metadata.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of **explicit** nonzeros (`nnz`).
    #[inline(always)]
    /// Details:
    /// - Purpose: Executes `nnz` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// True if the tensor stores no explicit nonzeros.
    #[inline(always)]
    /// Details:
    /// - Purpose: Checks whether `empty` condition is true.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ===================================================================
// ---------------------- Index Wrapping (toroidal) ------------------
// ===================================================================

/// Euclidean modulo for axis indices (supports negatives).
#[inline(always)]
/// Details:
/// - Purpose: Executes `wrap_axis_index` logic for this module.
/// - Parameters:
///   - `idx` (`isize`): Index argument selecting an element or slot.
///   - `dim` (`usize`): Parameter of type `usize` used by `wrap_axis_index`.
fn wrap_axis_index(idx: isize, dim: usize) -> usize {
    debug_assert!(dim > 0);
    let d = dim as isize;
    let mut m = idx % d;
    if m < 0 {
        m += d;
    }
    m as usize
}

// ===================================================================
// ----------------------------- Basics ------------------------------
// ===================================================================

impl<T: Scalar> Tensor<T> {
    /// Convert a multi-index (with negatives allowed) to a **row-major** flat index,
    /// using **per-axis periodic wrapping**.
    ///
    /// Same linearization as the dense tensor; this ensures interop is consistent.
    ///
    /// # Panics
    /// - Only if `idx.len() != self.shape.len()` (debug assertion).
    #[inline]
    /// Details:
    /// - Purpose: Computes an index mapping for input coordinates.
    /// - Parameters:
    ///   - `idx` (`&[isize]`): Index argument selecting an element or slot.
    pub fn index(&self, idx: &[isize]) -> usize {
        assert_eq!(idx.len(), self.shape.len(), "Index rank mismatch");
        let mut flat = 0usize;
        let mut stride = 1usize;
        for (&dim, &a_raw) in self.shape.iter().rev().zip(idx.iter().rev()) {
            let a = wrap_axis_index(a_raw, dim);
            flat += a * stride;
            stride *= dim;
        }
        flat
    }

    /// Get `Option<&T>` at multi-index (`None` if implicit zero).
    #[inline]
    /// Details:
    /// - Purpose: Returns the `opt` value.
    /// - Parameters:
    ///   - `idx` (`&[isize]`): Index argument selecting an element or slot.
    pub(crate) fn get_opt(&self, idx: &[isize]) -> Option<&T> {
        let k = self.index(idx);
        self.data.get(&k)
    }

    /// Get the value at multi-index, returning **zero** if absent.
    #[inline]
    /// Details:
    /// - Purpose: Executes `get` logic for this module.
    /// - Parameters:
    ///   - `idx` (`&[isize]`): Index argument selecting an element or slot.
    pub fn get(&self, idx: &[isize]) -> T {
        self.get_opt(idx).copied().unwrap_or_else(T::zero)
    }

    #[inline]
    /// Details:
    /// - Purpose: Returns the `mut_or_insert_zero` value.
    /// - Parameters:
    ///   - `idx` (`&[isize]`): Index argument selecting an element or slot.
    pub(crate) fn get_mut_or_insert_zero(&mut self, idx: &[isize]) -> &mut T {
        let k = self.index(idx);
        // Insert zero on miss, then return &mut to the stored value.
        self.data.entry(k).or_insert_with(T::zero)
    }

    /// Set value at multi-index. Inserting `0` **removes** the entry.
    ///
    /// This keeps the sparse invariant (no explicit zeros).
    #[inline]
    /// Details:
    /// - Purpose: Executes `set` logic for this module.
    /// - Parameters:
    ///   - `idx` (`&[isize]`): Index argument selecting an element or slot.
    ///   - `val` (`T`): Value provided by caller for write/update behavior.
    pub fn set(&mut self, idx: &[isize], val: T) {
        let k = self.index(idx);
        if val == T::zero() {
            self.data.remove(&k);
        } else {
            self.data.insert(k, val);
        }
    }

    /// Iterate over `(flat_index, &value)` of nonzeros.
    #[inline]
    /// Details:
    /// - Purpose: Executes `iter` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&usize, &T)> {
        self.data.iter()
    }

    /// **Internal helper**: build from `(flat_index, value)` pairs, dropping zeros.
    #[inline]
    /// Details:
    /// - Purpose: Builds this value from `flat_pairs` input.
    /// - Parameters:
    ///   - `shape` (`Vec<usize>`): Shape metadata defining tensor/grid dimensions.
    ///   - `pairs` (`Vec<(usize, T)>`): Parameter of type `Vec<(usize, T)>` used by `from_flat_pairs`.
    fn from_flat_pairs(shape: Vec<usize>, pairs: Vec<(usize, T)>) -> Self {
        let size = checked_num_elements(&shape, "sparse tensor from flat pairs");
        let mut map = AHashMap::with_capacity(pairs.len());
        for (k, v) in pairs {
            assert!(
                k < size,
                "sparse flat index out of bounds: {k} >= dense size {size}"
            );
            if v != T::zero() {
                map.insert(k, v);
            }
        }
        Self { shape, data: map }
    }
}

// ===================================================================
// ------------------------ Elementwise Ops --------------------------
// ===================================================================

/*
Elementwise binary ops (`+`, `-`, `*`, `/`) over **two** sparse tensors:

- We first compute the **union** of nonzero positions (flat keys).
- For each key, read `a` (default 0 if missing) and `b` (default 0).
- Apply the op, drop the result if it is zero.
- Construct the output with `from_flat_pairs`.

This avoids materializing dense intermediates and keeps sparsity.
*/

macro_rules! impl_sparse_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait for Tensor<T>
        where
            T: Scalar + $trait<Output = T> + Send + Sync,
        {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Tensor shape mismatch");

                // Union of keys (parallel sort + dedup).
                let mut keys: Vec<usize> =
                    Vec::with_capacity(self.data.len() + rhs.data.len());
                keys.extend(self.data.keys().copied());
                keys.extend(rhs.data.keys().copied());
                keys.par_sort_unstable();
                keys.dedup();

                let out_pairs: Vec<(usize, T)> = keys
                    .into_par_iter()
                    .filter_map(|k| {
                        let a = self.data.get(&k).copied().unwrap_or_else(T::zero);
                        let b = rhs.data.get(&k).copied().unwrap_or_else(T::zero);
                        let r = a $op b;
                        if r == T::zero() {
                            None
                        } else {
                            Some((k, r))
                        }
                    })
                    .collect();

                Self::from_flat_pairs(self.shape, out_pairs)
            }
        }
    };
}

impl_sparse_binop!(Add, add, +);
impl_sparse_binop!(Sub, sub, -);
impl_sparse_binop!(Mul, mul, *);
impl_sparse_binop!(Div, div, /);

// Optional: bitwise AND for integer-like types that support it.
// Uses **union** for simplicity (intersection would be a tiny optimization).
impl<T> BitAnd for Tensor<T>
where
    T: Scalar + BitAnd<Output = T> + Send + Sync,
{
    type Output = Self;

    #[inline]
    /// Details:
    /// - Purpose: Executes `bitand` logic for this module.
    /// - Parameters:
    ///   - `rhs` (`Self`): Parameter of type `Self` used by `bitand`.
    fn bitand(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape, "Tensor shape mismatch");

        let mut keys: Vec<usize> = Vec::with_capacity(self.data.len() + rhs.data.len());
        keys.extend(self.data.keys().copied());
        keys.extend(rhs.data.keys().copied());
        keys.par_sort_unstable();
        keys.dedup();

        let out_pairs: Vec<(usize, T)> = keys
            .into_par_iter()
            .filter_map(|k| {
                let a = self.data.get(&k).copied().unwrap_or_else(T::zero);
                let b = rhs.data.get(&k).copied().unwrap_or_else(T::zero);
                let r = a & b;
                if r == T::zero() { None } else { Some((k, r)) }
            })
            .collect();

        Self::from_flat_pairs(self.shape, out_pairs)
    }
}

// ===================================================================
// ------------------------ Scalar Ops (elem) ------------------------
// ===================================================================

/*
Elementwise ops with a **scalar RHS** (e.g., `S + c`, `S * c`).

We need to consume the hashmap by value to transform values. `AHashMap` by value
is not `IntoParallelIterator`, so we use **`.into_iter().par_bridge()`** to
bridge to rayon’s parallel pipeline. Zeros after the op are dropped.
*/

macro_rules! impl_sparse_scalar_binop_rhs_scalar {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait<T> for Tensor<T>
        where
            T: Scalar + $trait<Output = T> + Send + Sync,
        {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                let out_pairs: Vec<(usize, T)> = self
                    .data
                    .into_iter()
                    .par_bridge()
                    .map(|(k, v)| (k, v $op rhs))
                    .filter(|&(_, v)| v != T::zero())
                    .collect();

                Self::from_flat_pairs(self.shape, out_pairs)
            }
        }
    };
}

impl_sparse_scalar_binop_rhs_scalar!(Add, add, +);
impl_sparse_scalar_binop_rhs_scalar!(Sub, sub, -);
impl_sparse_scalar_binop_rhs_scalar!(Mul, mul, *);
impl_sparse_scalar_binop_rhs_scalar!(Div, div, /);

// ===================================================================
// ---------------------------- Type Casting -------------------------
// ===================================================================

impl<T: Scalar> Tensor<T> {
    /*
    Try to cast the sparse tensor into another scalar type `U`.

    - Real→Real or Complex→Complex: component-wise cast (re/im separately).
    - Real→Complex: imag part becomes 0.
    - Complex→Real: imag part is dropped (per project `Scalar` contract).
    - Zeros are automatically pruned.

    Returns an error if any component cannot be represented in `U::Real`.
    */

    /// Attempt an elementwise cast into `Tensor<U>` through `Scalar::try_cast`.
    pub fn try_cast_to<U: Scalar>(&self) -> Result<Tensor<U>, ScalarCastError> {
        let out_pairs: Result<Vec<(usize, U)>, _> = self
            .data
            .par_iter()
            .map(|(&k, &v)| v.try_cast::<U>().map(|u| (k, u)))
            .filter_map(|res| match res {
                Ok((k, v)) if v != U::zero() => Some(Ok((k, v))), // drop zeros
                Ok(_) => None,
                Err(e) => Some(Err(e)),
            })
            .collect();

        Ok(Tensor::<U>::from_flat_pairs(self.shape.clone(), out_pairs?))
    }

    /// Cast into `Tensor<U>`, **panicking** on failure.
    #[inline]
    pub fn cast_to<U: Scalar>(&self) -> Tensor<U> {
        self.try_cast_to::<U>()
            .expect("sparse tensor cast failed: component out of range for target type")
    }
}

// ===================================================================
// ---------------------- Convenience Constructors -------------------
// ===================================================================

impl<T: Scalar> Tensor<T> {
    /// Build from `(indices, value)` **triplets**; zeros are skipped.
    ///
    /// Note: The constructor is strict on bounds (no wrapping) to catch
    /// authoring mistakes; use runtime `set()` if you want wrapping.
    ///
    /// ```text
    /// 2x3 example, with entries at (0,1)=2, (1,2)=3:
    /// Tensor::<f64>::from_triplets(vec![2, 3], vec![(vec![0, 1], 2.0), (vec![1, 2], 3.0)])
    /// ```
    /// Details:
    /// - Purpose: Builds this value from `triplets` input.
    /// - Parameters:
    ///   - `shape` (`Vec<usize>`): Shape metadata defining tensor/grid dimensions.
    ///   - `triplets` (`impl IntoIterator<Item = (Vec<usize>, T)>`): Parameter of type `impl IntoIterator<Item = (Vec<usize>, T)>` used by `from_triplets`.
    pub fn from_triplets(
        shape: Vec<usize>,
        triplets: impl IntoIterator<Item = (Vec<usize>, T)>,
    ) -> Self {
        /// Details:
        /// - Purpose: Computes an index mapping for input coordinates.
        /// - Parameters:
        ///   - `shape` (`&[usize]`): Shape metadata defining tensor/grid dimensions.
        ///   - `idx` (`&[usize]`): Index argument selecting an element or slot.
        fn index_of(shape: &[usize], idx: &[usize]) -> usize {
            assert_eq!(idx.len(), shape.len(), "Triplet index rank mismatch");
            let mut flat = 0usize;
            let mut stride = 1usize;
            for (&dim, &a) in shape.iter().rev().zip(idx.iter().rev()) {
                assert!(a < dim, "Index out of bounds on an axis: {} >= {}", a, dim);
                flat += a * stride;
                stride *= dim;
            }
            flat
        }

        assert!(!shape.is_empty(), "Tensor rank must be >= 1");
        assert!(
            shape.iter().all(|&d| d > 0),
            "All dimensions must be > 0; got {shape:?}"
        );
        checked_num_elements(&shape, "sparse tensor from triplets");

        let mut map = AHashMap::default();
        for (idx, v) in triplets {
            if v == T::zero() {
                continue;
            }
            let k = index_of(&shape, &idx);
            map.insert(k, v);
        }
        Self { shape, data: map }
    }

    /// Convert to a **dense** tensor, allocating zeros for missing entries.
    ///
    /// Useful for debugging or interop with dense algorithms.
    #[inline]
    /// Details:
    /// - Purpose: Converts this value into `dense` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_dense(&self) -> TensorDense<T> {
        let size: usize = self.len_dense();
        let mut out = vec![T::zero(); size];
        for (&k, &v) in &self.data {
            out[k] = v;
        }
        TensorDense::from_parts_unchecked(self.shape.clone(), out)
    }

    /// Build a sparse tensor from a **dense** tensor by skipping zeros.
    #[inline]
    /// Details:
    /// - Purpose: Builds this value from `dense` input.
    /// - Parameters:
    ///   - `dense` (`&TensorDense<T>`): Parameter of type `&TensorDense<T>` used by `from_dense`.
    pub fn from_dense(dense: &TensorDense<T>) -> Self {
        let shape = dense.shape().to_vec();
        let size = checked_num_elements(&shape, "sparse tensor from dense");
        assert_eq!(size, dense.data().len(), "Dense size/shape mismatch");

        // Keep only nonzeros (indices are already row-major).
        let pairs: Vec<(usize, T)> = dense
            .data()
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(k, v)| if v == T::zero() { None } else { Some((k, v)) })
            .collect();

        Self::from_flat_pairs(shape, pairs)
    }

    /// Print a compact sparse summary and stored entries for direct sanity checks.
    pub fn print(&self) {
        println!(
            "Sparse tensor: shape={:?}, dense_size={}, nnz={}",
            self.shape,
            self.len_dense(),
            self.nnz()
        );

        if self.data.is_empty() {
            println!("  all entries are implicit zero");
            return;
        }

        let mut entries: Vec<(usize, T)> = self.data.iter().map(|(&k, &v)| (k, v)).collect();
        entries.par_sort_unstable_by_key(|&(k, _)| k);

        let shown = entries.len().min(32);
        for (k, value) in entries.iter().take(shown) {
            println!(
                "  [{:?}] flat={} value={}",
                self.unravel_index(*k),
                k,
                value
            );
        }

        if entries.len() > shown {
            println!("  ... {} more stored entries", entries.len() - shown);
        }
    }

    fn unravel_index(&self, flat: usize) -> Vec<usize> {
        let mut rem = flat;
        let mut idx = vec![0usize; self.shape.len()];
        for axis in (0..self.shape.len()).rev() {
            let dim = self.shape[axis];
            idx[axis] = rem % dim;
            rem /= dim;
        }
        idx
    }
}

// ===================================================================
// -------------------- TensorTrait Implementation -------------------
// ===================================================================

impl<T> TensorTrait<T> for Tensor<T>
where
    T: Scalar,
{
    type Repr<U: Scalar> = Tensor<U>;

    /// Create an **empty** sparse tensor with a given shape.
    ///
    /// # Panics
    /// Panics if `shape` is empty (rank 0) or contains a zero dimension.
    #[inline]
    /// Details:
    /// - Purpose: Executes `empty` logic for this module.
    /// - Parameters:
    ///   - `shape` (`&[usize]`): Shape metadata defining tensor/grid dimensions.
    fn empty(shape: &[usize]) -> Self {
        checked_num_elements(shape, "sparse tensor");
        Self {
            shape: shape.to_vec(),
            data: AHashMap::default(),
        }
    }

    /// Details:
    /// - Purpose: Returns the `sum` value.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn get_sum(&self) -> T {
        self.data
            .par_iter()
            .map(|(_, &x)| x)
            .reduce(|| T::zero(), |acc, x| acc + x)
    }

    /// Shape vector.
    #[inline]
    /// Details:
    /// - Purpose: Returns the logical shape metadata.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Row-major flat index with **per-axis periodic wrapping**.
    #[inline(always)]
    /// Details:
    /// - Purpose: Computes an index mapping for input coordinates.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): Parameter of type `&[isize]` used by `index`.
    fn index(&self, indices: &[isize]) -> usize {
        Tensor::<T>::index(self, indices)
    }

    /// Get by (wrapped) multi-index, returning zero if absent.
    #[inline(always)]
    /// Details:
    /// - Purpose: Executes `get` logic for this module.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): Parameter of type `&[isize]` used by `get`.
    fn get(&self, indices: &[isize]) -> T {
        Tensor::<T>::get(self, indices)
    }

    /// Sparse backend cannot safely yield `&mut T` via multi-index; Panic.
    #[inline(always)]
    /// Details:
    /// - Purpose: Returns the `mut` value.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): Parameter of type `&[isize]` used by `get_mut`.
    fn get_mut(&mut self, indices: &[isize]) -> &mut T {
        self.get_mut_or_insert_zero(indices)
    }

    /// Set value at (wrapped) multi-index (zero removes the entry).
    #[inline(always)]
    /// Details:
    /// - Purpose: Executes `set` logic for this module.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): Parameter of type `&[isize]` used by `set`.
    ///   - `val` (`T`): Value provided by caller for write/update behavior.
    fn set(&mut self, indices: &[isize], val: T) {
        Tensor::<T>::set(self, indices, val)
    }

    /// Parallel "fill": if `value == 0`, clears all entries; else sets all **existing**
    /// entries to `value` (keeps support but makes values uniform).
    #[inline]
    /// Details:
    /// - Purpose: Executes `par_fill` logic for this module.
    /// - Parameters:
    ///   - `value` (`T`): Value provided by caller for write/update behavior.
    fn par_fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        if value == T::zero() {
            self.data.clear();
            return;
        }

        // Only update existing nonzeros to the constant value.
        let keys: Vec<usize> = self.data.keys().copied().collect();
        let mut new_map = AHashMap::with_capacity(keys.len());
        for k in keys {
            new_map.insert(k, value);
        }
        self.data = new_map;
    }

    /// Parallel map-in-place over **existing** nonzeros; zeros after mapping are pruned.
    #[inline]
    fn par_map_in_place<F>(&mut self, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send,
    {
        // Clone pairs, map in parallel to (k, v'), drop zeros, rebuild.
        let pairs: Vec<(usize, T)> = self.iter().map(|(&k, &v)| (k, v)).collect();

        let mapped: Vec<(usize, T)> = pairs
            .into_par_iter()
            .map(|(k, v)| (k, f(v)))
            .filter(|&(_, v)| v != T::zero())
            .collect();

        self.data.clear();
        for (k, v) in mapped {
            self.data.insert(k, v);
        }
    }

    /// Parallel zip-with over **self's support** only, using `other.get(idx)` to read.
    #[inline]
    fn par_zip_with_inplace<F, Rhs>(&mut self, other: &Rhs, f: F)
    where
        Rhs: TensorTrait<T>,
        T: Copy + Send + Sync,
        F: Fn(T, T) -> T + Sync + Send,
    {
        assert_eq!(self.shape(), other.shape(), "Tensor shape mismatch");

        // Only iterate over current nonzeros in `self`.
        let rank = self.shape.len();
        let dims = self.shape.clone();

        let pairs: Vec<(usize, T)> = self.iter().map(|(&k, &v)| (k, v)).collect();

        let zipped: Vec<(usize, T)> = pairs
            .into_par_iter()
            .map(|(k, a)| {
                // linear -> multi-index (row-major)
                let mut rem = k;
                let mut idx = vec![0isize; rank];
                for ax in (0..rank).rev() {
                    let d = dims[ax];
                    idx[ax] = (rem % d) as isize;
                    rem /= d;
                }
                let b = other.get(&idx);
                (k, f(a, b))
            })
            .filter(|&(_, r)| r != T::zero())
            .collect();

        self.data.clear();
        for (k, v) in zipped {
            self.data.insert(k, v);
        }
    }

    /// Cast elementwise to another scalar type (panics on failure).
    #[inline]
    fn cast_to<U: Scalar + Send + Sync>(&self) -> Self::Repr<U>
    where
        T: Copy + Send + Sync,
    {
        self.try_cast_to::<U>()
            .expect("sparse tensor cast failed: component out of range for target type")
    }

    /// Details:
    /// - Purpose: Prints a human-readable representation.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn print(&self) {
        Tensor::<T>::print(self);
    }
}

// ===================================================================
// --------------------------- Extra Notes ---------------------------
// ===================================================================
//
// • Semantics:
//   - All axis indices are wrapped (toroidal); no OOB panics during access.
//   - Implicit zeros remain zero unless set; writing zero deletes the key.
//   - `get_mut()` by multi-index is unsupported for sparse and returns `None`.
//
// • Complexity:
//   - `index()`: O(rank).
//   - `get/set`: O(1) average (hash map) after index computation.
//   - Binary ops: O(nnz_a + nnz_b) + sorting cost for union keys.
//
// • Determinism:
//   - Wrapping semantics are deterministic for any `isize` (incl. large negatives).
// • Interop:
//   - `to_dense()` / `from_dense()` share the same row-major convention as `dense::Tensor`.
//
// • Testing hints:
//   - With shape [3,4], ensure `get([-1, -1]) == get([2, 3])`.
//   - Ensure `set([3, 0], v)` wraps to `[0, 0]`.
//

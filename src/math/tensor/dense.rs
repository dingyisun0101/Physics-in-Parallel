// src/math_foundations/tensor/dense.rs
/*! 
A **general-purpose N-dimensional dense tensor** backed by a flat `Vec<T>`.

Goals:
- **Performance-first**: contiguous memory layout with cache-friendly linear indexing.
- **Ergonomics**: safe multidimensional accessors, explicit raw/unsafe alternatives.
- **Parallelism**: `rayon`-powered in-place maps/zips and elementwise arithmetic.
- **Type-agnostic**: generic over a unified `Scalar` trait (real or complex).
- **Interoperability**: round-trip conversions to/from a sparse tensor type.

# Highlights

- `Tensor<T>::new(shape)`: zero-initialized tensor of shape `shape`.
- `index`, `get`, `get_mut`, `set`: multi-index access with **periodic wrapping** on each axis.
- **Negative indices are allowed** and are wrapped to the corresponding positive location.
- `get_by_idx*`, `set_by_idx`: linear-index access with **wrap-around** (toroidal) semantics.
- `par_fill`, `par_map_inplace`, `par_zip_with_inplace`: parallel in-place transforms.
- `Add/Sub/Mul/Div/BitAnd`: parallel elementwise binary ops with shape checks.
- `try_cast_to::<U>()` / `cast_to::<U>()`: whole-tensor scalar type conversion.
- `to_sparse()` / `from_sparse()`: dense↔sparse bridging.
- `ToString` + `from_string()`: simple textual roundtrip for small matrices.
- `print_2d()`: quick terminal visualization for 1D/2D tensors.

> **Note**  
> This file assumes a project-wide `Scalar` trait providing:
> - associated `type Real`
> - `fn re(&self) -> Self::Real`, `fn im(&self) -> Self::Real`
> - `fn from_re_im(r: Self::Real, i: Self::Real) -> Self`
> - `fn zero() -> Self`, `fn default() -> Self`
> and typical arithmetic traits. Adjust bounds if your `Scalar` differs.

> **Semantics (Important!)**  
> - All accessors use **toroidal wrapping**:
>   - Axis index `a` maps to `((a % dim) + dim) % dim` (Euclidean modulo).
>   - Linear index `k` maps to `k % len`.
> - Therefore, **no accessor ever panics on index** (except rank-mismatch in `index()` via debug-assert).
> - These semantics are ideal for lattice/periodic-boundary simulations.

*/

use std::fmt::Display;
use std::ops::{Add, Sub, Mul, Div, BitAnd};
use std::str::FromStr;

use rayon::prelude::*;
use serde::Serialize;
use num_traits::NumCast;

use super::super::scalar::Scalar;
use super::sparse::Tensor as TensorSparse;
use super::tensor_trait::TensorTrait;

//===================================================================
// -------------------------- Basic Struct --------------------------
//===================================================================

/// A dense N-D tensor with row-major (C-style) linearization.
///
/// - Elements are stored in a single contiguous `Vec<T>` in row-major order.
/// - Shape is a `Vec<usize>` where `shape.len()` is the rank, and
///   `shape.iter().product()` equals the number of elements.
///
/// # Invariants
/// - `data.len() == shape.iter().product()`.
#[derive(Debug, Clone, Serialize)]
pub struct Tensor<T: Scalar> {
    /// The extents along each axis. Example: `[rows, cols]` for 2D.
    pub shape: Vec<usize>,
    /// Flat, row-major storage of all elements.
    pub data: Vec<T>,
}

impl<T: Scalar> Tensor<T> {
    /// Create a new tensor with the given `shape`, filled with `T::default()`.
    ///
    /// # Panics
    /// Panics if `shape` contains a zero dimension or if `product` overflows `usize`.
    #[inline]
    pub fn new(shape: Vec<usize>) -> Self {
        assert!(
            shape.iter().all(|&d| d > 0),
            "All dimensions must be > 0; got {shape:?}"
        );
        let size = shape.iter().product::<usize>();
        Self {
            shape,
            data: vec![T::default(); size],
        }
    }

    /// Number of elements (a.k.a. linear size).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True iff there are zero elements (never true given our shape assertion).
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

//===================================================================
// ------------------------ Index Wrapping --------------------------
//===================================================================

/// Euclidean modulo for axis indices (supports negatives).
#[inline(always)]
fn wrap_axis_index(idx: isize, dim: usize) -> usize {
    debug_assert!(dim > 0);
    let d = dim as isize;
    let mut m = idx % d;
    if m < 0 { m += d; }
    m as usize
}

/// Wrap linear index into `[0, len)`.
#[inline(always)]
fn wrap_linear_index(idx: usize, len: usize) -> usize {
    debug_assert!(len > 0);
    idx % len
}

//===================================================================
// ------------------------- Raw Accessors --------------------------
//===================================================================
// These methods operate on **linear** indices and always return
// a valid reference by wrapping the index into range.

impl<T: Scalar> Tensor<T> {
    #[inline(always)]
    pub fn get_by_idx(&self, idx: usize) -> &T {
        let k = wrap_linear_index(idx, self.data.len());
        // SAFETY: k < len by construction
        unsafe { self.data.get_unchecked(k) }
    }

    #[inline(always)]
    pub fn get_by_idx_mut(&mut self, idx: usize) -> &mut T {
        let k = wrap_linear_index(idx, self.data.len());
        // SAFETY: k < len by construction
        unsafe { self.data.get_unchecked_mut(k) }
    }

    #[inline(always)]
    pub fn set_by_idx(&mut self, idx: usize, val: T) {
        let k = wrap_linear_index(idx, self.data.len());
        // SAFETY: k < len by construction
        unsafe { *self.data.get_unchecked_mut(k) = val }
    }
}

//===================================================================
// ------------------------ Tensor Trait Impl -----------------------
//===================================================================

impl<T> TensorTrait<T> for Tensor<T>
where
    T: Scalar,
{
    type Repr<U: Scalar> = Tensor<U>;

    /// Shape vector.
    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Row-major linearization with **per-axis periodic wrapping**.
    ///
    /// - Accepts negative indices and arbitrarily large/small signed values.
    /// - Never panics due to out-of-bounds (only rank mismatch is debug-asserted).
    #[inline(always)]
    fn index(&self, indices: &[isize]) -> usize {
        debug_assert_eq!(indices.len(), self.shape.len(), "Index rank mismatch");

        // Compute flat index by accumulating a * stride.
        // We iterate from the last axis to the first to build the stride.
        let mut flat = 0usize;
        let mut stride = 1usize;

        for (&dim, &raw_a) in self.shape.iter().rev().zip(indices.iter().rev()) {
            let a = wrap_axis_index(raw_a, dim);
            flat += a * stride;
            stride *= dim;
        }
        flat
    }

    /// Get by (wrapped) multi-index. Returns a copy of T (Scalar assumed Copy).
    #[inline(always)]
    fn get(&self, indices: &[isize]) -> T {
        let k = self.index(indices);
        // SAFETY: k is wrapped into [0, len)
        unsafe { *self.data.get_unchecked(k) }
    }

    /// Get mutable reference by (wrapped) multi-index.
    /// Returns `Some(&mut T)` (always `Some` with current semantics).
    #[inline(always)]
    fn get_mut(&mut self, indices: &[isize]) -> Option<&mut T> {
        let k = self.index(indices);
        // SAFETY: k is wrapped into [0, len)
        Some(unsafe { self.data.get_unchecked_mut(k) })
    }

    /// Set value at (wrapped) multi-index.
    #[inline(always)]
    fn set(&mut self, indices: &[isize], val: T) {
        let k = self.index(indices);
        // SAFETY: k is wrapped into [0, len)
        unsafe { *self.data.get_unchecked_mut(k) = val }
    }

    /// Parallel fill with a constant value.
    #[inline]
    fn par_fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.data.par_iter_mut().for_each(|x| *x = value);
    }

    /// Parallel in-place map with a pure function.
    #[inline]
    fn par_map_in_place<F>(&mut self, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send,
    {
        self.data.par_iter_mut().for_each(|x| *x = f(*x));
    }

    /// Parallel in-place zip with another tensor-like structure.
    ///
    /// This calls `other.get(&idx)` for each linear position `k`,
    /// converting `k` to a row-major multi-index `idx`.
    #[inline]
    fn par_zip_with_inplace<F, Rhs>(&mut self, other: &Rhs, f: F)
    where
        Rhs: TensorTrait<T> + ?Sized,
        T: Copy + Send + Sync,
        F: Fn(T, T) -> T + Sync + Send,
    {
        let rank = self.shape.len();
        let dims = self.shape.clone();

        self.data
            .par_iter_mut()
            .enumerate()
            .for_each(|(k, a)| {
                // linear -> multi-index (row-major)
                let mut rem = k;
                let mut idx = vec![0isize; rank];
                for ax in (0..rank).rev() {
                    let d = dims[ax];
                    // `rem % d` is in [0, d); convert to isize (non-negative)
                    idx[ax] = (rem % d) as isize;
                    rem /= d;
                }
                let b = other.get(&idx);
                *a = f(*a, b);
            });
    }

    /// Eager, element-wise type cast (panics on failure).
    #[inline]
    fn cast_to<U: Scalar>(&self) -> Self::Repr<U>
    where
        T: Scalar,
    {
        // Call the inherent `try_cast_to` to avoid trait recursion.
        self.try_cast_to::<U>()
            .expect("tensor cast failed: component out of range for target type")
    }
}

//===================================================================
// ------------------------- Arithmetic Ops -------------------------
//===================================================================

macro_rules! impl_tensor_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait for Tensor<T>
        where
            T: Scalar + $trait<Output = T> + Send + Sync + Copy,
        {
            type Output = Self;

            /// Parallel elementwise binary op with shape check.
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Tensor shape mismatch");
                let shape = self.shape;
                let a = self.data;
                let b = rhs.data;

                // NOTE: into_par_iter() consumes; map is parallel and lock-free.
                let data = a
                    .into_par_iter()
                    .zip(b.into_par_iter())
                    .map(|(x, y)| x $op y)
                    .collect();

                Self { shape, data }
            }
        }
    };
}

impl_tensor_op!(Add, add, +);
impl_tensor_op!(Sub, sub, -);
impl_tensor_op!(Mul, mul, *);
impl_tensor_op!(Div, div, /);
impl_tensor_op!(BitAnd, bitand, &);

// ===================================================================
// ---------------------------- Type Casting -------------------------
// ===================================================================

impl<T: Scalar> Tensor<T> {
    /// Attempt to cast `self` (component-wise) into `Tensor<U>`.
    /// Returns an error if any component over/underflows or cannot be represented.
    ///
    /// - Complex→Real: uses `.re()`/`.im()` pair and reconstructs `U` via `from_re_im`.
    /// - Parallelized over elements.
    pub fn try_cast_to<U: Scalar>(&self) -> Result<Tensor<U>, &'static str> {
        #[inline(always)]
        fn cast_scalar<T: Scalar, U: Scalar>(x: T) -> Result<U, &'static str> {
            let r_t: T::Real = x.re();
            let i_t: T::Real = x.im();

            let r_u: U::Real =
                NumCast::from(r_t).ok_or("real part out of range for target type")?;
            let i_u: U::Real =
                NumCast::from(i_t).ok_or("imag part out of range for target type")?;

            Ok(U::from_re_im(r_u, i_u))
        }

        let data: Result<Vec<U>, _> = self
            .data
            .par_iter()
            .map(|&x| cast_scalar::<T, U>(x))
            .collect();

        Ok(Tensor {
            shape: self.shape.clone(),
            data: data?,
        })
    }
}

// ===================================================================
// ---------------------- Convenience Constructors -------------------
// ===================================================================

impl<T: Scalar + Display> ToString for Tensor<T> {
    /// Serializes as nested bracket blocks with `;` separators, e.g.:
    /// `[[1;2;3];[4;5;6]]` for a 2×3 tensor.
    fn to_string(&self) -> String {
        fn format_recursive<T: Display>(data: &[T], shape: &[usize]) -> String {
            if shape.len() == 1 {
                let mut line = String::new();
                for i in 0..shape[0] {
                    line.push_str(&format!("{}", data[i]));
                    if i + 1 != shape[0] {
                        line.push(';');
                    }
                }
                line
            } else {
                let stride = shape[1..].iter().product::<usize>();
                let mut result = String::new();
                for i in 0..shape[0] {
                    let start = i * stride;
                    let end = start + stride;
                    let nested = format_recursive(&data[start..end], &shape[1..]);
                    result.push('[');
                    result.push_str(&nested);
                    result.push(']');
                    if i + 1 != shape[0] {
                        result.push(';');
                    }
                }
                result
            }
        }

        format!("[{}]", format_recursive(&self.data, &self.shape))
    }
}

impl<T: Scalar + FromStr> Tensor<T> {
    /// Parse a small tensor from a bracketed `;`-separated string produced by `ToString`.
    ///
    /// # Panics
    /// Panics on malformed input or inconsistent row lengths.
    pub fn from_string(s: &str) -> Self {
        let normalized = s
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .replace("];", "]|");

        let rows: Vec<Vec<T>> = normalized
            .split('|')
            .map(|row| {
                row.replace('[', "")
                    .replace(']', "")
                    .split(';')
                    .filter(|x| !x.trim().is_empty())
                    .map(|num| {
                        let cleaned = num.trim();
                        cleaned.parse::<T>().unwrap_or_else(|_| {
                            panic!("Invalid number in tensor: '{}'", cleaned)
                        })
                    })
                    .collect()
            })
            .collect();

        let num_rows = rows.len();
        assert!(num_rows > 0, "Tensor cannot be empty");

        let num_cols = rows[0].len();
        assert!(
            rows.iter().all(|r| r.len() == num_cols),
            "All rows must have the same number of columns"
        );

        let flat = rows.into_iter().flatten().collect();

        Tensor {
            shape: vec![num_rows, num_cols],
            data: flat,
        }
    }
}

impl<T: Scalar> Tensor<T> {
    /// Convert this **dense** tensor to a **sparse** one by skipping zeros.
    #[inline]
    pub fn to_sparse(&self) -> TensorSparse<T> {
        TensorSparse::from_dense(self)
    }

    /// Build a **dense** tensor from a **sparse** one (missing entries = zero).
    #[inline]
    pub fn from_sparse(sparse: &TensorSparse<T>) -> Self {
        let shape = sparse.shape().to_vec();
        let size: usize = shape.iter().product();
        let mut data = vec![T::zero(); size];

        for (&k, &v) in sparse.iter() {
            // SAFETY: k < size as guaranteed by the sparse structure.
            unsafe { *data.get_unchecked_mut(k) = v; }
        }

        Self { shape, data }
    }
}

//===================================================================
// -------------------------- Utilities -----------------------------
//===================================================================

impl<T: Scalar + Display + Copy> Tensor<T> {
    /// Quick-and-dirty printer for 1D/2D tensors to stdout.
    ///
    /// # Panics
    /// Panics if `rank > 2`.
    pub fn print_2d(&self) {
        match self.shape.len() {
            1 => {
                for i in 0..self.shape[0] {
                    print!("{:<8} ", self.get(&[i as isize]));
                }
                println!();
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];
                for i in 0..rows {
                    for j in 0..cols {
                        print!("{:<8} ", self.get(&[i as isize, j as isize]));
                    }
                    println!();
                }
            }
            _ => panic!("print_2d only supports rank 1 or 2 tensors"),
        }
    }
}

// ------------------------------------------------------------------
// Back-compat convenience: keep the old spelling around if you call
// `par_map_inplace` directly on dense tensors outside the trait.
// ------------------------------------------------------------------
impl<T> Tensor<T>
where
    T: Scalar + Copy + Send + Sync,
{
    #[inline]
    pub fn par_map_inplace<F>(&mut self, f: F)
    where
        F: Fn(T) -> T + Sync + Send,
    {
        <Self as TensorTrait<T>>::par_map_in_place(self, f);
    }
}

//===================================================================
// --------------------------- Extra Notes --------------------------
//===================================================================
//
// • Complexity:
//   - `index()`: O(rank), very small constant factors.
//   - `get/set`: O(1) after index computation; use raw accessors if you pre-linearize.
//   - All parallel ops are data-parallel with rayon; choose chunk sizes via rayon global config.
//
// • Safety:
//   - All internal reads/writes use `get_unchecked` after **wrapping** the index; thus safe.
//   - The only potential panic is a **debug assertion** on index rank mismatch.
//
// • Determinism:
//   - Wrapping semantics are deterministic for any `isize` index (including extreme negatives).
//   - Linear wrapping is `idx % len` which matches typical circular buffer behavior.
//
// • Interop:
//   - If you need **clamped** or **panic-on-OOB** behavior for certain algorithms,
//     consider adding alternate accessors (`get_clamped`, `get_strict`) alongside these.
//
// • Testing tips:
//   - Ensure `get([-1, -1]) == get([shape[0]-1, shape[1]-1])`.
//   - Ensure `get([shape[0] as isize, 0]) == get([0, 0])`.
//   - For linear: `get_by_idx(len) == get_by_idx(0)`; `get_by_idx(usize::MAX)` is valid and wraps.
//

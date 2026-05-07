// src/math_foundations/tensor/dense.rs
/*!
A **general-purpose N-dimensional dense tensor** backed by a flat `Vec<T>`.

Goals:
- **Performance-first**: contiguous memory layout with cache-friendly linear indexing.
- **Ergonomics**: safe multidimensional accessors through the tensor facade.
- **Parallelism**: `rayon`-powered in-place maps/zips and elementwise arithmetic.
- **Type-agnostic**: generic over the crate-wide `Scalar` trait (real or complex).
- **Computation-only scope**: JSON, ndarray, and string interop live under `math::io`.

# Highlights

- `Tensor<T>::empty(shape)`: zero-initialized tensor of shape `shape`.
- `index`, `get`, `get_mut`, `set`: multi-index access with **periodic wrapping** on each axis.
- **Negative indices are allowed** and are wrapped to the corresponding positive location.
- `par_fill`, `par_map_inplace`, `par_zip_with_inplace`: parallel in-place transforms.
- `Add/Sub/Mul/Div/BitAnd`: parallel elementwise binary ops with shape checks.
- `try_cast_to::<U>()` / `cast_to::<U>()`: whole-tensor scalar type conversion.
- `to_sparse()` / `from_sparse()`: dense↔sparse bridging.
- `print()`: quick terminal visualization, choosing a compact presentation by rank.

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
> - Therefore, **no accessor ever panics on bounds**; rank mismatches panic explicitly.
> - These semantics are ideal for lattice/periodic-boundary simulations.

*/

use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use rayon::prelude::*;

use super::errors;
use super::sparse::Tensor as TensorSparse;
use super::tensor_trait::TensorTrait;
use crate::math::scalar::{Scalar, ScalarCastError};

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
#[derive(Debug, Clone)]
pub struct Tensor<T: Scalar> {
    /// The extents along each axis. Example: `[rows, cols]` for 2D.
    pub(crate) shape: Vec<usize>,
    /// Flat, row-major storage of all elements.
    pub(crate) data: Vec<T>,
}

impl<T: Scalar> Tensor<T> {
    /// Number of elements (a.k.a. linear size).
    #[inline(always)]
    /// Details:
    /// - Purpose: Returns the current length/size.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True iff there are zero elements (never true given our shape assertion).
    #[inline(always)]
    /// Details:
    /// - Purpose: Checks whether `empty` condition is true.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub(crate) fn from_parts_unchecked(shape: Vec<usize>, data: Vec<T>) -> Self {
        Self { shape, data }
    }

    #[inline]
    pub(crate) fn from_vec(shape: &[usize], data: Vec<T>) -> Self {
        let expected = checked_num_elements(shape, "dense tensor from vector");
        assert_eq!(
            data.len(),
            expected,
            "dense tensor data length mismatch: expected {expected}, got {}",
            data.len()
        );
        Self {
            shape: shape.to_vec(),
            data,
        }
    }

    #[inline]
    pub(crate) fn data(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub(crate) fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

//===================================================================
// ------------------------ Index Wrapping --------------------------
//===================================================================

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

pub(crate) fn checked_num_elements(shape: &[usize], context: &str) -> usize {
    errors::checked_num_elements(shape).unwrap_or_else(|error| panic!("{context}: {error}"))
}

//===================================================================
// ------------------------ Tensor Trait Impl -----------------------
//===================================================================

impl<T> TensorTrait<T> for Tensor<T>
where
    T: Scalar,
{
    type Repr<U: Scalar> = Tensor<U>;

    /// Create a new tensor with the given `shape`, filled with `T::default()`.
    ///
    /// # Panics
    /// Panics if `shape` contains a zero dimension or if `product` overflows `usize`.
    #[inline]
    /// Details:
    /// - Purpose: Executes `empty` logic for this module.
    /// - Parameters:
    ///   - `shape` (`&[usize]`): Shape metadata defining tensor/grid dimensions.
    fn empty(shape: &[usize]) -> Self {
        assert!(
            shape.iter().all(|&d| d > 0),
            "All dimensions must be > 0; got {shape:?}"
        );
        let size = checked_num_elements(shape, "dense tensor");
        Self {
            shape: shape.to_vec(),
            data: vec![T::default(); size],
        }
    }

    /// Details:
    /// - Purpose: Returns the `sum` value.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn get_sum(&self) -> T {
        let result = self
            .data
            .par_iter()
            .cloned()
            .reduce(|| T::zero(), |a, b| a + b);
        result
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

    /// Row-major linearization with **per-axis periodic wrapping**.
    ///
    /// - Accepts negative indices and arbitrarily large/small signed values.
    /// - Never panics due to out-of-bounds (only rank mismatch is debug-asserted).
    #[inline(always)]
    /// Details:
    /// - Purpose: Computes an index mapping for input coordinates.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): Parameter of type `&[isize]` used by `index`.
    fn index(&self, indices: &[isize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Index rank mismatch");

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
    /// Details:
    /// - Purpose: Executes `get` logic for this module.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): Parameter of type `&[isize]` used by `get`.
    fn get(&self, indices: &[isize]) -> T {
        let k = self.index(indices);
        // SAFETY: k is wrapped into [0, len)
        unsafe { *self.data.get_unchecked(k) }
    }

    /// Get mutable reference by (wrapped) multi-index.
    /// Returns `Some(&mut T)` (always `Some` with current semantics).
    #[inline(always)]
    /// Details:
    /// - Purpose: Returns the `mut` value.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): Parameter of type `&[isize]` used by `get_mut`.
    fn get_mut(&mut self, indices: &[isize]) -> &mut T {
        let k = self.index(indices);
        // SAFETY: k is wrapped into [0, len)
        unsafe { self.data.get_unchecked_mut(k) }
    }

    /// Set value at (wrapped) multi-index.
    #[inline(always)]
    /// Details:
    /// - Purpose: Executes `set` logic for this module.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): Parameter of type `&[isize]` used by `set`.
    ///   - `val` (`T`): Value provided by caller for write/update behavior.
    fn set(&mut self, indices: &[isize], val: T) {
        let k = self.index(indices);
        // SAFETY: k is wrapped into [0, len)
        unsafe { *self.data.get_unchecked_mut(k) = val }
    }

    /// Parallel fill with a constant value.
    #[inline]
    /// Details:
    /// - Purpose: Executes `par_fill` logic for this module.
    /// - Parameters:
    ///   - `value` (`T`): Value provided by caller for write/update behavior.
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
        Rhs: TensorTrait<T>,
        T: Copy + Send + Sync,
        F: Fn(T, T) -> T + Sync + Send,
    {
        assert_eq!(self.shape(), other.shape(), "Tensor shape mismatch");
        let rank = self.shape.len();
        let dims = self.shape.clone();

        self.data.par_iter_mut().enumerate().for_each(|(k, a)| {
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

    /// Fallible, element-wise type cast.
    #[inline]
    fn try_cast_to<U: Scalar>(&self) -> Result<Self::Repr<U>, ScalarCastError>
    where
        T: Copy + Send + Sync,
    {
        Tensor::<T>::try_cast_to::<U>(self)
    }

    /// Details:
    /// - Purpose: Prints a human-readable representation.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn print(&self) {
        Tensor::<T>::print(self);
    }
}

//===================================================================
// ------------------------- Arithmetic Ops -------------------------
//===================================================================
// ------------------------ &Tensor ⊕ &Tensor -> Tensor ------------------------

macro_rules! impl_tensor_ref_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> $trait<&'a Tensor<T>> for &'a Tensor<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<Output = T>,
        {
            type Output = Tensor<T>;
            #[inline]
            fn $method(self, rhs: &'a Tensor<T>) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Tensor shape mismatch");
                let mut out = self.clone(); // reuses lhs allocation
                out.data
                    .par_iter_mut()
                    .zip(rhs.data.par_iter())
                    .for_each(|(a, &b)| { *a = *a $op b; });
                out
            }
        }
    };
}
impl_tensor_ref_binop!(Add, add, +);
impl_tensor_ref_binop!(Sub, sub, -);
impl_tensor_ref_binop!(Mul, mul, *);
impl_tensor_ref_binop!(Div, div, /);

// ------------------------ Tensor ⊕= &Tensor (in-place) -----------------------

macro_rules! impl_tensor_ref_assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> $trait<&'a Tensor<T>> for Tensor<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<T>,
        {
            #[inline]
            fn $method(&mut self, rhs: &'a Tensor<T>) {
                assert_eq!(self.shape, rhs.shape, "Tensor shape mismatch");
                self.data
                    .par_iter_mut()
                    .zip(rhs.data.par_iter())
                    .for_each(|(a, &b)| { *a = (*a) $op b; });
            }
        }
    };
}
impl_tensor_ref_assign!(AddAssign, add_assign, +);
impl_tensor_ref_assign!(SubAssign, sub_assign, -);
impl_tensor_ref_assign!(MulAssign, mul_assign, *);
impl_tensor_ref_assign!(DivAssign, div_assign, /);

// ------------------------ &Tensor ⊕ scalar -> Tensor -------------------------

macro_rules! impl_tensor_ref_scalar_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> $trait<T> for &'a Tensor<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<Output = T>,
        {
            type Output = Tensor<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                let mut out = self.clone();
                out.data.par_iter_mut().for_each(|a| *a = *a $op rhs);
                out
            }
        }
    };
}
impl_tensor_ref_scalar_binop!(Add, add, +);
impl_tensor_ref_scalar_binop!(Sub, sub, -);
impl_tensor_ref_scalar_binop!(Mul, mul, *);
impl_tensor_ref_scalar_binop!(Div, div, /);

// ------------------------ Tensor ⊕= scalar (in-place) ------------------------

macro_rules! impl_tensor_scalar_assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait<T> for Tensor<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<T>,
        {
            #[inline]
            fn $method(&mut self, rhs: T) {
                self.data.par_iter_mut().for_each(|a| *a = *a $op rhs);
            }
        }
    };
}
impl_tensor_scalar_assign!(AddAssign, add_assign, +);
impl_tensor_scalar_assign!(SubAssign, sub_assign, -);
impl_tensor_scalar_assign!(MulAssign, mul_assign, *);
impl_tensor_scalar_assign!(DivAssign, div_assign, /);

// ===================================================================
// ---------------------------- Type Casting -------------------------
// ===================================================================

impl<T: Scalar> Tensor<T> {
    /// Attempt to cast `self` elementwise into `Tensor<U>`.
    /// Returns an error if any component over/underflows or cannot be represented.
    ///
    /// - Each element is converted through `Scalar::try_cast`.
    /// - Parallelized over elements.
    pub fn try_cast_to<U: Scalar>(&self) -> Result<Tensor<U>, ScalarCastError> {
        let data: Result<Vec<U>, _> = self.data.par_iter().map(|&x| x.try_cast::<U>()).collect();

        Ok(Tensor {
            shape: self.shape.clone(),
            data: data?,
        })
    }
}

// ===================================================================
// ---------------------- Convenience Constructors -------------------
// ===================================================================

impl<T: Scalar> Tensor<T> {
    /// Convert this **dense** tensor to a **sparse** one by skipping zeros.
    #[inline]
    /// Details:
    /// - Purpose: Converts this value into `sparse` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_sparse(&self) -> TensorSparse<T> {
        TensorSparse::from_dense(self)
    }

    /// Build a **dense** tensor from a **sparse** one (missing entries = zero).
    #[inline]
    /// Details:
    /// - Purpose: Builds this value from `sparse` input.
    /// - Parameters:
    ///   - `sparse` (`&TensorSparse<T>`): Parameter of type `&TensorSparse<T>` used by `from_sparse`.
    pub fn from_sparse(sparse: &TensorSparse<T>) -> Self {
        let shape = sparse.shape().to_vec();
        let size = checked_num_elements(&shape, "dense tensor from sparse");
        let mut data = vec![T::zero(); size];

        for (&k, &v) in sparse.iter() {
            // SAFETY: k < size as guaranteed by the sparse structure.
            unsafe {
                *data.get_unchecked_mut(k) = v;
            }
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
    /// Details:
    /// - Purpose: Prints a human-readable representation.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn print(&self) {
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
            _ => {
                println!(
                    "Tensor shape {:?}, {} elements",
                    self.shape,
                    self.data.len()
                );
                println!("{}", crate::math::io::string::format_dense_storage(self));
            }
        }
    }
}

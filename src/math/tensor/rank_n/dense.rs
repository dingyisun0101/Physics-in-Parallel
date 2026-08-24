// src/math_foundations/tensor/dense.rs
/*!
A **general-purpose N-dimensional dense tensor** backed by a flat `Vec<T>`.

Goals:
- **Performance-first**: contiguous memory layout with cache-friendly linear indexing.
- **Ergonomics**: safe multidimensional accessors through the tensor facade.
- **Parallelism**: `rayon`-powered in-place maps/zips and elementwise arithmetic.
- **Type-agnostic**: generic over the crate-wide `Scalar` trait (real or complex).
- **Computation-only scope**: JSON and string interop live under `math::io`.

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

use crate::threading::{parallel_chunk_len, should_parallelize_elements};
use rayon::prelude::*;

use super::errors;
use super::layout::{flat_index_wrapped, normalize_flat_index};
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
#[derive(Debug, Clone, PartialEq)]
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
    /// - Purpose: Returns the total number of scalar entries stored in the
    ///   dense row-major buffer.
    /// - Parameters:
    ///   - (none): Uses this tensor's stored shape and data buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True iff there are zero elements (never true given our shape assertion).
    #[inline(always)]
    /// Details:
    /// - Purpose: Reports whether the dense backing buffer contains no scalar
    ///   entries. For tensors constructed through validated constructors this
    ///   is normally false because rank-zero and zero-axis shapes are rejected.
    /// - Parameters:
    ///   - (none): Uses this tensor's backing buffer length.
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
    /// - Purpose: Allocates a dense tensor with the requested shape and fills
    ///   every stored scalar entry with `T::default()`.
    /// - Parameters:
    ///   - `shape` (`&[usize]`): Non-empty list of positive axis lengths that
    ///     defines the tensor rank and dense storage size.
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
    /// - Purpose: Computes the scalar sum of all entries in the dense buffer
    ///   using a Rayon reduction.
    /// - Parameters:
    ///   - (none): Sums every logical tensor element.
    fn sum(&self) -> T {
        self.data
            .par_iter()
            .with_min_len(parallel_chunk_len(self.data.len()).unwrap_or(1))
            .cloned()
            .reduce(|| T::zero(), |a, b| a + b)
    }

    /// Shape vector.
    #[inline]
    /// Details:
    /// - Purpose: Returns the tensor's axis lengths without exposing mutable
    ///   access to the internal shape vector.
    /// - Parameters:
    ///   - (none): Reads this tensor's stored shape metadata.
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Row-major linearization with **per-axis periodic wrapping**.
    ///
    /// - Accepts negative indices and arbitrarily large/small signed values.
    /// - Never panics due to out-of-bounds (only rank mismatch is debug-asserted).
    #[inline(always)]
    ///
    /// Details:
    /// - Purpose: Converts a full multidimensional coordinate into the
    ///   corresponding row-major flat buffer index, applying periodic wrapping
    ///   independently on every axis.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): One signed coordinate per tensor axis; the
    ///     slice length must match the tensor rank.
    fn index(&self, indices: &[isize]) -> usize {
        flat_index_wrapped(&self.shape, indices).expect("Index rank mismatch")
    }

    /// Get by (wrapped) multi-index. Returns a copy of T (Scalar assumed Copy).
    #[inline(always)]
    /// Details:
    /// - Purpose: Reads and returns a copy of the scalar at the wrapped
    ///   multidimensional coordinate.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): One signed coordinate per axis; negative and
    ///     oversized coordinates wrap periodically.
    fn get(&self, indices: &[isize]) -> T {
        let k = self.index(indices);
        // SAFETY: k is wrapped into [0, len)
        unsafe { *self.data.get_unchecked(k) }
    }

    #[inline(always)]
    fn get_flat(&self, index: isize) -> T {
        let index = normalize_flat_index(index, self.data.len());
        // SAFETY: the normalized index is within the nonempty dense buffer.
        unsafe { *self.data.get_unchecked(index) }
    }

    /// Get mutable reference by (wrapped) multi-index.
    /// Returns `Some(&mut T)` (always `Some` with current semantics).
    #[inline(always)]
    /// Details:
    /// - Purpose: Returns a mutable reference to the scalar at the wrapped
    ///   multidimensional coordinate so callers can update it in place.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): One signed coordinate per axis; negative and
    ///     oversized coordinates wrap periodically.
    fn get_mut(&mut self, indices: &[isize]) -> &mut T {
        let k = self.index(indices);
        // SAFETY: k is wrapped into [0, len)
        unsafe { self.data.get_unchecked_mut(k) }
    }

    #[inline(always)]
    fn get_flat_mut(&mut self, index: isize) -> &mut T {
        let index = normalize_flat_index(index, self.data.len());
        // SAFETY: the normalized index is within the nonempty dense buffer.
        unsafe { self.data.get_unchecked_mut(index) }
    }

    /// Set value at (wrapped) multi-index.
    #[inline(always)]
    /// Details:
    /// - Purpose: Replaces the scalar stored at the wrapped multidimensional
    ///   coordinate with the caller-provided value.
    /// - Parameters:
    ///   - `indices` (`&[isize]`): One signed coordinate per axis; negative and
    ///     oversized coordinates wrap periodically.
    ///   - `val` (`T`): New scalar value to store at that coordinate.
    fn set(&mut self, indices: &[isize], val: T) {
        let k = self.index(indices);
        // SAFETY: k is wrapped into [0, len)
        unsafe { *self.data.get_unchecked_mut(k) = val }
    }

    #[inline(always)]
    fn set_flat(&mut self, index: isize, val: T) {
        *self.get_flat_mut(index) = val;
    }

    /// Parallel fill with a constant value.
    #[inline]
    /// Details:
    /// - Purpose: Replaces every dense-buffer entry with the same scalar value
    ///   using Rayon over the contiguous storage.
    /// - Parameters:
    ///   - `value` (`T`): Scalar copied into every logical tensor element.
    fn par_fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        if !should_parallelize_elements(self.data.len()) {
            self.data.fill(value);
            return;
        }
        let min_elements_per_job = parallel_chunk_len(self.data.len()).unwrap_or(1);
        self.data
            .par_iter_mut()
            .with_min_len(min_elements_per_job)
            .for_each(|x| *x = value);
    }

    /// Parallel in-place map with a pure function.
    #[inline]
    fn par_map_in_place<F>(&mut self, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send,
    {
        if !should_parallelize_elements(self.data.len()) {
            self.data.iter_mut().for_each(|value| *value = f(*value));
            return;
        }
        let min_elements_per_job = parallel_chunk_len(self.data.len()).unwrap_or(1);
        self.data
            .par_iter_mut()
            .with_min_len(min_elements_per_job)
            .for_each(|x| *x = f(*x));
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
        if !should_parallelize_elements(self.data.len()) {
            self.data
                .iter_mut()
                .enumerate()
                .for_each(|(index, value)| *value = f(*value, other.get_flat(index as isize)));
            return;
        }
        let min_elements_per_job = parallel_chunk_len(self.data.len()).unwrap_or(1);
        self.data
            .par_iter_mut()
            .with_min_len(min_elements_per_job)
            .enumerate()
            .for_each(|(k, a)| {
                let b = other.get_flat(k as isize);
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
    /// - Purpose: Satisfies the tensor trait printing contract by delegating to
    ///   the dense tensor's inherent rank-aware terminal printer.
    /// - Parameters:
    ///   - (none): Reads this tensor without modifying it.
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
                    .with_min_len(parallel_chunk_len(rhs.data.len()).unwrap_or(1))
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
                    .with_min_len(parallel_chunk_len(rhs.data.len()).unwrap_or(1))
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
                let min_elements_per_job = parallel_chunk_len(out.data.len()).unwrap_or(1);
                out.data
                    .par_iter_mut()
                    .with_min_len(min_elements_per_job)
                    .for_each(|a| *a = *a $op rhs);
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
                let min_elements_per_job = parallel_chunk_len(self.data.len()).unwrap_or(1);
                self.data
                    .par_iter_mut()
                    .with_min_len(min_elements_per_job)
                    .for_each(|a| *a = *a $op rhs);
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
        let data: Result<Vec<U>, _> = self
            .data
            .par_iter()
            .with_min_len(parallel_chunk_len(self.data.len()).unwrap_or(1))
            .map(|&x| x.try_cast::<U>())
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

impl<T: Scalar> Tensor<T> {
    /// Convert this **dense** tensor to a **sparse** one by skipping zeros.
    #[inline]
    /// Details:
    /// - Purpose: Creates an equivalent sparse tensor by preserving the shape
    ///   and storing only entries whose value is not `T::zero()`.
    /// - Parameters:
    ///   - (none): Reads this dense tensor's full buffer.
    pub fn to_sparse(&self) -> TensorSparse<T> {
        TensorSparse::from_dense(self)
    }

    /// Build a **dense** tensor from a **sparse** one (missing entries = zero).
    #[inline]
    /// Details:
    /// - Purpose: Densifies a sparse tensor by allocating the full row-major
    ///   buffer and writing sparse entries into a zero-filled tensor.
    /// - Parameters:
    ///   - `sparse` (`&TensorSparse<T>`): Sparse tensor whose shape and stored
    ///     nonzero values should be represented densely.
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
    /// - Purpose: Prints a compact terminal representation chosen by rank:
    ///   one-line output for rank 1, table output for rank 2, and structured
    ///   nested output for higher ranks.
    /// - Parameters:
    ///   - (none): Reads this tensor without modifying it.
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

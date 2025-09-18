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
- `get`, `get_mut`, `set`: safe multi-index read/write.
- `get_by_idx*`, `set_by_idx`: fast linear-index access (with bounds assert + unchecked).
- `par_fill`, `par_map_inplace`, `par_zip_with_inplace`: parallel in-place transforms.
- `Add/Sub/Mul/Div/BitAnd`: parallel elementwise binary ops with shape checks.
- `try_cast_to::<U>()` / `cast_to::<U>()`: whole-tensor scalar type conversion.
- `to_sparse()` / `from_sparse()`: dense↔sparse bridging.
- `ToString` + `from_string()`: simple textual roundtrip for small matrices.
- `print_2d()`: quick terminal visualization for 1D/2D tensors.

> **Note**  
> This file assumes a project-wide `Scalar` trait providing:
> - associated `type Real: num_traits::Num + ...`
> - `fn re(&self) -> Self::Real`, `fn im(&self) -> Self::Real`
> - `fn from_re_im(r: Self::Real, i: Self::Real) -> Self`
> - `fn zero() -> Self`, `fn default() -> Self`
> and typical arithmetic traits. Adjust bounds if your `Scalar` differs.

*/

use std::fmt::Display;
use std::ops::{Add, Sub, Mul, Div, BitAnd};
use std::str::FromStr;

use rayon::prelude::*;
use serde::Serialize;
use num_traits::NumCast;

use super::super::scalar::Scalar;
use super::sparse::Tensor as TensorSparse;

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
        // Defensive: forbid zero-sized axes. If you want to support them, remove this.
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
}

//===================================================================
// ------------------------- Raw Accessors --------------------------
//===================================================================
// These methods trade ergonomics for performance: they operate on **linear**
// indices and use `get_unchecked` after explicit bound assertions. Prefer the
// safe multi-index versions below unless you’re in a hot path and know what
// you’re doing.

impl<T: Scalar> Tensor<T> {
    /// Borrow the element at **linear** index `idx`.
    ///
    /// # Safety model
    /// Asserts bounds once, then uses `get_unchecked` for speed.
    #[inline(always)]
    pub fn get_by_idx(&self, idx: usize) -> &T {
        assert!(idx < self.data.len(), "Index {} out of bounds", idx);
        unsafe { self.data.get_unchecked(idx) }
    }

    /// Mutably borrow the element at **linear** index `idx`.
    ///
    /// # Safety model
    /// Asserts bounds once, then uses `get_unchecked_mut` for speed.
    #[inline(always)]
    pub fn get_by_idx_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.data.len(), "Index {} out of bounds", idx);
        unsafe { self.data.get_unchecked_mut(idx) }
    }

    /// Set the value at **linear** index `idx` to `val`.
    ///
    /// # Safety model
    /// Asserts bounds once, then uses `get_unchecked_mut`.
    #[inline(always)]
    pub fn set_by_idx(&mut self, idx: usize, val: T) {
        assert!(idx < self.data.len(), "Index {} out of bounds", idx);
        unsafe { *self.data.get_unchecked_mut(idx) = val }
    }
}

//===================================================================
// ----------------------- Parallel Iterator ------------------------
//===================================================================
// In-place parallel transforms backed by `rayon`. These mutate `self`.

impl<T: Scalar> Tensor<T> {
    /// **Parallel fill**: set every element to the given copyable `value`.
    #[inline]
    pub fn par_fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.data.par_iter_mut().for_each(|x| *x = value);
    }

    /// **Parallel elementwise map (in-place)** using a pure function `f`.
    ///
    /// `f` should be side-effect-free and cheap to clone.  
    /// `T: Copy` is required to read-by-value while writing in place.
    #[inline]
    pub fn par_map_inplace<F>(&mut self, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send,
    {
        self.data.par_iter_mut().for_each(|x| *x = f(*x));
    }

    /// **Parallel zip-with (in-place)**: `self[i] = f(self[i], other[i])`.
    ///
    /// # Panics
    /// Panics on shape mismatch.
    #[inline]
    pub fn par_zip_with_inplace<F>(&mut self, other: &Tensor<T>, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T, T) -> T + Sync + Send,
    {
        assert_eq!(self.data.len(), other.data.len(), "shape mismatch");
        self.data
            .par_iter_mut()
            .zip(other.data.par_iter())
            .for_each(|(a, &b)| *a = f(*a, b));
    }
}

//===================================================================
// --------------------------- Accessors ----------------------------
//===================================================================
// Safe multidimensional accessors convert `[i,j,k,...]` into a flat index.

impl<T: Scalar> Tensor<T> {
    /// Compute the **row-major linear index** for a multi-index `indices`.
    ///
    /// This method checks rank and in-bounds for each axis.
    ///
    /// # Panics
    /// - If `indices.len() != self.shape.len()`.
    /// - If any index is out of bounds.
    #[inline(always)]
    pub fn index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Index rank mismatch");
        // Row-major: last axis is fastest.
        let mut idx = 0usize;
        let mut stride = 1usize;
        for (i, &dim) in self.shape.iter().rev().enumerate() {
            let axis = self.shape.len() - 1 - i;
            let aidx = indices[axis];
            assert!(aidx < dim, "Index out of bounds on axis {}: {} !< {}", axis, aidx, dim);
            idx += aidx * stride;
            stride *= dim;
        }
        idx
    }

    /// Borrow the element at multi-index `indices`.
    ///
    /// # Panics
    /// If rank or any coordinate is out of bounds.
    #[inline(always)]
    pub fn get(&self, indices: &[usize]) -> &T {
        let idx = self.index(indices);
        unsafe { self.data.get_unchecked(idx) }
    }

    /// Mutably borrow the element at multi-index `indices`.
    ///
    /// # Panics
    /// If rank or any coordinate is out of bounds.
    #[inline(always)]
    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let idx = self.index(indices);
        unsafe { self.data.get_unchecked_mut(idx) }
    }

    /// Set the value at multi-index `indices` to `val`.
    ///
    /// # Panics
    /// If rank or any coordinate is out of bounds.
    #[inline(always)]
    pub fn set(&mut self, indices: &[usize], val: T) {
        let idx = self.index(indices);
        unsafe { *self.data.get_unchecked_mut(idx) = val }
    }
}

//===================================================================
// ------------------------- Arithmetic Ops -------------------------
//===================================================================
// Elementwise binary ops producing a new tensor. Shapes must match.

macro_rules! impl_tensor_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait for Tensor<T>
        where
            T: Scalar + $trait<Output = T> + Send + Sync,
        {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Tensor shape mismatch");
                // Move fields out of `self` and `rhs` to avoid partial move issues.
                let shape = self.shape;
                let a = self.data;
                let b = rhs.data;

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
    /* 
    Try to cast the entire tensor into another scalar type `U`.

    - Real→Real or Complex→Complex: component-wise cast via `NumCast` on (re, im).
    - Real→Complex: imag part becomes 0.
    - Complex→Real: imag part is dropped (per project `Scalar` contract).
    - Fails if any component is not representable in `U::Real`.

    This function is parallelized and returns a `Result`.
    */

    /// Attempt to cast `self` (component-wise) into `Tensor<U>`.
    /// Returns an error if any component over/underflows or cannot be represented.
    pub fn try_cast_to<U: Scalar>(&self) -> Result<Tensor<U>, &'static str> {
        #[inline(always)]
        fn cast_scalar<T: Scalar, U: Scalar>(x: T) -> Result<U, &'static str> {
            let r_t: T::Real = x.re();
            let i_t: T::Real = x.im();

            // Cast real and imaginary parts into U::Real
            let r_u: U::Real =
                NumCast::from(r_t).ok_or("real part out of range for target type")?;
            let i_u: U::Real =
                NumCast::from(i_t).ok_or("imag part out of range for target type")?;

            // Build the target value (imag ignored for real targets)
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

    /// Cast the entire tensor into another scalar type `U`, **panicking** on failure.
    ///
    /// Use this when you’re sure the cast cannot overflow/underflow.
    #[inline]
    pub fn cast_to<U: Scalar>(&self) -> Tensor<U> {
        self.try_cast_to::<U>()
            .expect("tensor cast failed: component out of range for target type")
    }
}

// ===================================================================
// ---------------------- Convenience Constructors -------------------
// ===================================================================

impl<T: Scalar + Display> ToString for Tensor<T> {
    /// Serialize small-rank tensors to a compact bracketed string.
    ///
    /// Format examples:
    /// - 1D: `[a;b;c]`
    /// - 2D: `[[a;b];[c;d]]`
    ///
    /// This is meant for quick debugging/roundtrip, not for large tensors.
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
    /// Parse a string like `"[[1;2;3];[4;5;6]]"` into a 2D tensor (or nested ranks).
    ///
    /// This is a **simple helper** intended for small, rectangular inputs.
    /// It strips outer brackets, treats `;` as a separator, and supports nested lists.
    ///
    /// # Panics
    /// - On malformed input or non-rectangular shapes.
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
    ///
    /// Delegates to `TensorSparse::from_dense(self)` to keep logic in one place.
    #[inline]
    pub fn to_sparse(&self) -> TensorSparse<T> {
        TensorSparse::from_dense(self)
    }

    /// Build a **dense** tensor from a **sparse** one (missing entries = zero).
    ///
    /// # Notes
    /// - Relies on the sparse tensor to provide valid flat indices.
    /// - Fills unmentioned entries with `T::zero()`.
    #[inline]
    pub fn from_sparse(sparse: &TensorSparse<T>) -> Self {
        let shape = sparse.shape().to_vec();
        let size: usize = shape.iter().product();
        let mut data = vec![T::zero(); size];

        // Fill explicit nonzeros. Iteration is over (&usize, &T).
        for (&k, &v) in sparse.iter() {
            // Safety: `k` must be < size (guaranteed by sparse invariant).
            unsafe { *data.get_unchecked_mut(k) = v; }
        }

        Self { shape, data }
    }
}

//===================================================================
// -------------------------- Utilities -----------------------------
//===================================================================

impl<T: Scalar + Display> Tensor<T> {
    /// Quick-and-dirty printer for 1D/2D tensors to stdout.
    ///
    /// # Panics
    /// Panics if `rank > 2`.
    pub fn print_2d(&self) {
        match self.shape.len() {
            1 => {
                for i in 0..self.shape[0] {
                    print!("{:<8} ", self.get(&[i]));
                }
                println!();
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];
                for i in 0..rows {
                    for j in 0..cols {
                        print!("{:<8} ", self.get(&[i, j]));
                    }
                    println!();
                }
            }
            _ => panic!("print_2d only supports rank 1 or 2 tensors"),
        }
    }
}
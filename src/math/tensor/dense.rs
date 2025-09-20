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
- `index`, `get`, `get_mut`, `set`: multi-index access **with negative indices supported**.
- `get_by_idx*`, `set_by_idx`: fast linear-index access (with bounds assert + unchecked).
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
}

//===================================================================
// ------------------------- Raw Accessors --------------------------
//===================================================================
// These methods operate on **linear** indices and use `get_unchecked` after
// explicit bound assertions.

impl<T: Scalar> Tensor<T> {
    #[inline(always)]
    pub fn get_by_idx(&self, idx: usize) -> &T {
        assert!(idx < self.data.len(), "Index {} out of bounds", idx);
        unsafe { self.data.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn get_by_idx_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.data.len(), "Index {} out of bounds", idx);
        unsafe { self.data.get_unchecked_mut(idx) }
    }

    #[inline(always)]
    pub fn set_by_idx(&mut self, idx: usize, val: T) {
        assert!(idx < self.data.len(), "Index {} out of bounds", idx);
        unsafe { *self.data.get_unchecked_mut(idx) = val }
    }
}

//===================================================================
// ------------------------ Index Normalization ---------------------
//===================================================================

#[inline(always)]
fn norm_axis_index(idx: isize, dim: usize) -> usize {
    // Support negative indices: -1 → dim-1, -2 → dim-2, ...
    if idx >= 0 {
        let u = idx as usize;
        assert!(u < dim, "Index out of bounds on axis (>=0): {} !< {}", u, dim);
        u
    } else {
        // Avoid overflow on isize::MIN by doing (-(idx+1)) + 1
        let abs = (-(idx + 1)) as usize + 1;
        assert!(abs <= dim, "Index out of bounds on axis (<0): -{} > dim {}", abs, dim);
        dim - abs
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

    #[inline(always)]
    fn index(&self, indices: &[isize]) -> usize {
        debug_assert_eq!(indices.len(), self.shape.len(), "Index rank mismatch");
        // Row-major linearization with negative-index support.
        let mut flat = 0usize;
        let mut stride = 1usize;
        for (rev_axis, &dim) in self.shape.iter().rev().enumerate() {
            let axis = self.shape.len() - 1 - rev_axis;
            let raw = indices[axis];
            let a = if raw >= 0 {
                let u = raw as usize;
                assert!(u < dim, "Index out of bounds on axis {}: {} >= {}", axis, u, dim);
                u
            } else {
                let abs = (-(raw + 1)) as usize + 1;
                assert!(abs <= dim, "Index out of bounds on axis {}: -{} > {}", axis, abs, dim);
                dim - abs
            };
            flat += a * stride;
            stride *= dim;
        }
        flat
    }

    #[inline(always)]
    fn get(&self, indices: &[isize]) -> T {
        let k = self.index(indices);
        unsafe { *self.data.get_unchecked(k) }
    }

    #[inline(always)]
    fn get_mut(&mut self, indices: &[isize]) -> Option<&mut T> {
        let k = self.index(indices);
        Some(unsafe { self.data.get_unchecked_mut(k) })
    }

    #[inline(always)]
    fn set(&mut self, indices: &[isize], val: T) {
        let k = self.index(indices);
        unsafe { *self.data.get_unchecked_mut(k) = val }
    }

    #[inline]
    fn par_fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.data.par_iter_mut().for_each(|x| *x = value);
    }

    #[inline]
    fn par_map_in_place<F>(&mut self, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send,
    {
        self.data.par_iter_mut().for_each(|x| *x = f(*x));
    }

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
                    idx[ax] = (rem % d) as isize;
                    rem /= d;
                }
                let b = other.get(&idx);
                *a = f(*a, b);
            });
    }

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
            T: Scalar + $trait<Output = T> + Send + Sync,
        {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Tensor shape mismatch");
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
    /// Attempt to cast `self` (component-wise) into `Tensor<U>`.
    /// Returns an error if any component over/underflows or cannot be represented.
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

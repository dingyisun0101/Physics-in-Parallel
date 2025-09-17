// src/math_foundations/tensor/dense.rs
/* 
    A general-purpose N-dimensional tensor backed by a flat vector.
    This struct is designed for high performance and parallel operations.
    It supports arbitrary shapes and provides methods for safe and unsafe access.
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

#[derive(Debug, Clone, Serialize)]
pub struct Tensor<T: Scalar> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T: Scalar > Tensor<T> {
    /// Create a new tensor with given shape, filled with default values.
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            shape,
            data: vec![T::default(); size],
        }
    }
}

//===================================================================
// ------------------------- Raw Accessors --------------------------
//===================================================================
// These methods provide direct access to the underlying data
// without bounds checking. Use with caution!

impl<T: Scalar> Tensor<T> {
    /// Get a reference to the element at linear index `idx`
    #[inline(always)]
    pub fn get_by_idx(&self, idx: usize) -> &T {
        assert!(idx < self.data.len(), "Index {} out of bounds", idx);
        unsafe { self.data.get_unchecked(idx) }
    }

    /// Get a mutable reference to the element at linear index `idx`
    #[inline(always)]
    pub fn get_by_idx_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.data.len(), "Index {} out of bounds", idx);
        unsafe { self.data.get_unchecked_mut(idx) }
    }

    /// Set the value at linear index `idx`
    #[inline(always)]
    pub fn set_by_idx(&mut self, idx: usize, val: T) {
        assert!(idx < self.data.len(), "Index {} out of bounds", idx);
        unsafe { *self.data.get_unchecked_mut(idx) = val }
    }
}





//===================================================================
// ----------------------- Parallel Iterator ------------------------
//===================================================================

impl<T: Scalar> Tensor<T> {
    /// Parallel fill with a copyable value.
    #[inline]
    pub fn par_fill(&mut self, value: T)
    where
        T: Copy,
    {
        self.data.par_iter_mut().for_each(|x| *x = value);
    }

    /// Parallel elementwise map (in-place).
    #[inline]
    pub fn par_map_inplace<F>(&mut self, f: F)
    where
        T: Copy,
        F: Fn(T) -> T + Sync + Send,
    {
        self.data.par_iter_mut().for_each(|x| *x = f(*x));
    }

    /// Parallel elementwise zip-with assignment: self[i] = f(self[i], other[i]).
    #[inline]
    pub fn par_zip_with_inplace<F>(&mut self, other: &Tensor<T>, f: F)
    where
        T: Copy,
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
// These methods provide safe access to tensor elements using multi-dimensional indices.
// They perform bounds checking and ensure the indices match the tensor's shape.

impl<T: Scalar> Tensor<T> {
    #[inline(always)]
    pub fn index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Index rank mismatch");

        let mut idx = 0;
        let mut stride = 1;
        for (i, &dim) in self.shape.iter().rev().enumerate() {
            let axis = self.shape.len() - 1 - i;
            assert!(indices[axis] < dim, "Index out of bounds");
            idx += indices[axis] * stride;
            stride *= dim;
        }
        idx
    }

    #[inline(always)]
    pub fn get(&self, indices: &[usize]) -> &T {
        let idx = self.index(indices);
        unsafe { self.data.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let idx = self.index(indices);
        unsafe { self.data.get_unchecked_mut(idx) }
    }

    #[inline(always)]
    pub fn set(&mut self, indices: &[usize], val: T) {
        let idx = self.index(indices);
        unsafe { *self.data.get_unchecked_mut(idx) = val }
    }
}

//===================================================================
// ------------------------- Arithmetic Ops -------------------------
//===================================================================
// Implement arithmetic operations for tensors using traits.
// These operations require tensors to have the same shape.

macro_rules! impl_tensor_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait for Tensor<T>
        where
            T: Scalar + $trait<Output = T>,
        {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Tensor shape mismatch");

                let data = self
                    .data
                    .into_par_iter()
                    .zip(rhs.data.into_par_iter())
                    .map(|(a, b)| a $op b)
                    .collect();

                Self {
                    shape: self.shape,
                    data,
                }
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
// ---------------------------- Type Casting --------------------------
// ===================================================================

impl<T: Scalar> Tensor<T> {
    /* 
        Try to cast the entire tensor into another scalar type `U`.
        Real→Real or Complex→Complex: component-wise cast.
        Real→Complex: imag part becomes 0.
        Complex→Real: imag part dropped (per `Scalar::from_re_im` contract).
        Fails if any component cannot be represented in the target type.
    */

    pub fn try_cast_to<U: Scalar>(&self) -> Result<Tensor<U>, &'static str> {
        #[inline(always)]
        fn cast_scalar<T: Scalar, U: Scalar>(x: T) -> Result<U, &'static str> {
            let r_t: T::Real = x.re();
            let i_t: T::Real = x.im();

            // Cast real and imaginary parts into U::Real
            let r_u: U::Real = NumCast::from(r_t).ok_or("real part out of range for target type")?;
            let i_u: U::Real = NumCast::from(i_t).ok_or("imag part out of range for target type")?;

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

    /// Cast the entire tensor into another scalar type `U`, panicking on failure.
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
    /// Parse a string like "[[1;2;3];[4;5;6];[7;8;9]]" into a 2D tensor.
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
    /// Convert this dense tensor to a sparse one by skipping zeros.
    #[inline]
    pub fn to_sparse(&self) -> TensorSparse<T> {
        // Reuse the sparse helper (keeps code in one place).
        TensorSparse::from_dense(self)
    }

    /// Build a dense tensor from a sparse one (missing entries filled with zero).
    #[inline]
    pub fn from_sparse(sparse: &TensorSparse<T>) -> Self {
        let shape = sparse.shape().to_vec();
        let size: usize = shape.iter().product();
        let mut data = vec![T::zero(); size];

        // Fill explicit nonzeros. Iteration is over (&usize, &T).
        for (&k, &v) in sparse.iter() {
            // Safety: `k` is always < size because `TensorSparse` only stores valid flats.
            unsafe { *data.get_unchecked_mut(k) = v; }
        }

        Self { shape, data }
    }
}


//===================================================================
// -------------------------- Utilities -----------------------------
//===================================================================

impl<T: Scalar + Display> Tensor<T> {
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
            _ => panic!("print_tensor only supports 1D or 2D tensors"),
        }
    }
}
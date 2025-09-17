// src/math_foundations/tensor/sparse.rs
/* 
    A hash-backed sparse N-D tensor.
        - Storage: `AHashMap<flat_index, T>`; zeros are implicit (not stored).
        - `T` must be your unified `Scalar` (ints, floats, Complex).
        - Elementwise ops work on the **union** of nonzero indices; zeros are dropped.
        - Parallelized with `rayon` for binary ops; scalar ops use `par_bridge()`.
*/


use ahash::AHashMap;
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;
use rayon::iter::ParallelBridge;
use std::ops::{Add, Sub, Mul, Div, BitAnd};
use num_traits::NumCast;

use super::super::scalar::Scalar;
use super::dense::Tensor as TensorDense;



// ===================================================================
// --------------------------- Struct Def ----------------------------
// ===================================================================

#[derive(Clone, Debug)]
pub struct Tensor<T: Scalar> {
    shape: Vec<usize>,
    data: AHashMap<usize, T>, // flat index -> value (non-zero)
}





// ===================================================================
// ----------------------------- Basics ------------------------------
// ===================================================================

impl<T: Scalar> Tensor<T> {
    /// Create an empty sparse tensor with a given shape.
    pub fn new(shape: Vec<usize>) -> Self {
        assert!(!shape.is_empty(), "Tensor rank must be >= 1");
        Self { shape, data: AHashMap::default() }
    }

    /// Rank (number of dimensions).
    #[inline] pub fn rank(&self) -> usize { self.shape.len() }

    /// Shape vector.
    #[inline] pub fn shape(&self) -> &[usize] { &self.shape }

    /// Number of explicitly stored entries (≠ 0).
    #[inline] pub fn nnz(&self) -> usize { self.data.len() }

    /// True if no nonzero is stored.
    #[inline] pub fn is_empty(&self) -> bool { self.data.is_empty() }

    /// Row-major flat index from multi-index. Panics if out of bounds.
    #[inline]
    pub fn index(&self, idx: &[usize]) -> usize {
        assert_eq!(idx.len(), self.shape.len(), "Index rank mismatch");
        let mut flat = 0usize;
        let mut stride = 1usize;
        for (rev_axis, &dim) in self.shape.iter().rev().enumerate() {
            let axis = self.shape.len() - 1 - rev_axis;
            let a = idx[axis];
            assert!(a < dim, "Index out of bounds on axis {}: {} >= {}", axis, a, dim);
            flat += a * stride;
            stride *= dim;
        }
        flat
    }

    /// Get `Option<&T>` at multi-index (None if implicit zero).
    #[inline]
    pub fn get_opt(&self, idx: &[usize]) -> Option<&T> {
        let k = self.index(idx);
        self.data.get(&k)
    }

    /// Get the value at multi-index, returning zero if absent.
    #[inline]
    pub fn get(&self, idx: &[usize]) -> T {
        self.get_opt(idx).copied().unwrap_or_else(T::zero)
    }

    /// Set value at multi-index. `0` removes the entry.
    #[inline]
    pub fn set(&mut self, idx: &[usize], val: T) {
        let k = self.index(idx);
        if val == T::zero() { self.data.remove(&k); }
        else { self.data.insert(k, val); }
    }

    /// Add (accumulate) `delta` into entry at multi-index, then prune zero.
    #[inline]
    pub fn add_assign_at(&mut self, idx: &[usize], delta: T)
    where
        T: Add<Output = T>
    {
        if delta == T::zero() { return; }
        let k = self.index(idx);
        let newv = match self.data.get(&k).copied() {
            Some(v) => v + delta,
            None => delta,
        };
        if newv == T::zero() { self.data.remove(&k); } else { self.data.insert(k, newv); }
    }

    /// Internal helper: build from (flat_index, value) pairs, dropping zeros.
    fn from_flat_pairs(shape: Vec<usize>, pairs: Vec<(usize, T)>) -> Self {
        let mut map = AHashMap::with_capacity(pairs.len());
        for (k, v) in pairs {
            if v != T::zero() { map.insert(k, v); }
        }
        Self { shape, data: map }
    }

    /// Iterate over (flat_index, &value) of nonzeros.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &T)> {
        self.data.iter()
    }

    /// Unsafe (no bounds check) set by flat index. 0 removes.
    #[inline]
    pub fn set_by_flat(&mut self, k: usize, val: T) {
        if val == T::zero() { self.data.remove(&k); } else { self.data.insert(k, val); }
    }

    /// Get by flat index with zero default.
    #[inline]
    pub fn get_by_flat(&self, k: usize) -> T {
        self.data.get(&k).copied().unwrap_or_else(T::zero)
    }
}





// ===================================================================
// ------------------------ Elementwise Ops --------------------------
// ===================================================================

macro_rules! impl_sparse_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait for Tensor<T>
        where
            T: Scalar + $trait<Output = T>,
        {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "Tensor shape mismatch");

                // union of keys
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
                        let r = a $op b;
                        if r == T::zero() { None } else { Some((k, r)) }
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
impl<T> BitAnd for Tensor<T>
where
    T: Scalar + BitAnd<Output = T>
{
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape, "Tensor shape mismatch");

        // Use union for simplicity; intersection could be a micro-optimization.
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

macro_rules! impl_sparse_scalar_binop_rhs_scalar {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait<T> for Tensor<T>
        where
            T: Scalar + $trait<Output = T>
        {
            type Output = Self;

            fn $method(self, rhs: T) -> Self::Output {
                // `AHashMap` doesn't implement `IntoParallelIterator` by value.
                // Consume it serially and bridge to rayon for parallel map/filter.
                let out_pairs: Vec<(usize, T)> = self.data
                    .into_iter()
                    .par_bridge()
                    .map(|(k, v)| (k, v $op rhs))
                    .filter(|&(_, v)| v != T::zero())
                    .collect();

                Self::from_flat_pairs(self.shape, out_pairs)
            }
        }
    }
}

impl_sparse_scalar_binop_rhs_scalar!(Add, add, +);
impl_sparse_scalar_binop_rhs_scalar!(Sub, sub, -);
impl_sparse_scalar_binop_rhs_scalar!(Mul, mul, *);
impl_sparse_scalar_binop_rhs_scalar!(Div, div, /);





// ===================================================================
// ---------------------------- Type Casting --------------------------
// ===================================================================

impl<T: Scalar> Tensor<T> {
    /*
        Try to cast the sparse tensor into another scalar type `U`.
        Real→Real or Complex→Complex: component-wise cast.
        Real→Complex: imag part becomes 0.
        Complex→Real: imag part dropped (per `Scalar::from_re_im` contract).
        Zeros are automatically pruned.
     */
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

        let out_pairs: Result<Vec<(usize, U)>, _> = self
            .data
            .par_iter()
            .map(|(&k, &v)| {
                cast_scalar::<T, U>(v).map(|u| (k, u))
            })
            .filter_map(|res| match res {
                Ok((k, v)) if v != U::zero() => Some(Ok((k, v))),
                Ok(_) => None, // drop zeros
                Err(e) => Some(Err(e)),
            })
            .collect();

        Ok(Tensor::<U>::from_flat_pairs(self.shape.clone(), out_pairs?))
    }

    /// Cast the sparse tensor into another scalar type `U`, panicking on failure.
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
    /// Build from (indices, value) triplets; zeros are skipped.
    pub fn from_triplets(
        shape: Vec<usize>,
        triplets: impl IntoIterator<Item = (Vec<usize>, T)>
    ) -> Self {
        fn index_of(shape: &[usize], idx: &[usize]) -> usize {
            assert_eq!(idx.len(), shape.len(), "Triplet index rank mismatch");
            let mut flat = 0usize;
            let mut stride = 1usize;
            for (rev_axis, &dim) in shape.iter().rev().enumerate() {
                let axis = shape.len() - 1 - rev_axis;
                let a = idx[axis];
                assert!(a < dim, "Index out of bounds on axis {}: {} >= {}", axis, a, dim);
                flat += a * stride;
                stride *= dim;
            }
            flat
        }

        let mut map = AHashMap::default();
        for (idx, v) in triplets {
            if v == T::zero() { continue; }
            let k = index_of(&shape, &idx);
            map.insert(k, v);
        }
        Self { shape, data: map }
    }

    /// Convert to a dense flat vector (row-major), allocating zeros for missing entries.
    /// Useful for debugging or interop.
    pub fn to_dense(&self) -> TensorDense<T> {
        let size: usize = self.shape.iter().product();
        let mut out = vec![T::zero(); size];
        for (&k, &v) in &self.data {
            out[k] = v;
        }
        TensorDense { shape: (self.shape.clone()), data: (out) }
    }

    /// Build from a dense tensor by skipping zeros.
    pub fn from_dense(dense: &TensorDense<T>) -> Self {
        let shape = dense.shape.clone();
        let size: usize = shape.iter().product();
        debug_assert_eq!(size, dense.data.len(), "Dense size/shape mismatch");

        // Collect only non-zeros (keep flat indices as-is since dense is row-major).
        let pairs: Vec<(usize, T)> = dense.data
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(k, v)| if v == T::zero() { None } else { Some((k, v)) })
            .collect();

        Self::from_flat_pairs(shape, pairs)
    }
}

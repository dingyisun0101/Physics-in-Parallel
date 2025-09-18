// src/math_foundations/vector_list.rs
/*!
A **Structure-of-Arrays (SoA)** wrapper over the dense `Tensor<T>` type to represent
a list of fixed-length vectors. The underlying `Tensor` has shape **[D, n]** where:

- `D` is the (runtime) vector dimensionality.
- `n` is the number of vectors in the list (runtime).

This layout stores all 0-th components together, then all 1-st components, etc.
*/

use rayon::prelude::*;
use serde::Serialize;

use super::super::scalar::Scalar;
use super::super::tensor::dense::Tensor;

// ============================================================================
// --------------------------- Struct Def: SoA --------------------------------
// ============================================================================

/// A list of `n` vectors in **Structure-of-Arrays** form backed by `Tensor<T>`.
/// Invariant: `data.shape == [D, n]` with both dimensions > 0 allowed (n may be 0).
#[derive(Debug, Clone, Serialize)]
pub struct VectorList<T: Scalar> {
    /// Underlying dense storage; must always have shape `[D, n]`.
    pub data: Tensor<T>,
}

/// Mutable proxy to one vector (column index `i`) exposing D disjoint `&mut T`.
pub struct VecRefMut<'a, T: Scalar> {
    cols: Vec<&'a mut T>,
}

impl<'a, T: Scalar> VecRefMut<'a, T> {
    #[inline] pub fn len(&self) -> usize { self.cols.len() }

    #[inline] pub fn get(&self, k: usize) -> &T {
        debug_assert!(k < self.cols.len());
        self.cols[k]
    }

    #[inline] pub fn get_mut(&mut self, k: usize) -> &mut T {
        debug_assert!(k < self.cols.len());
        self.cols[k]
    }

    #[inline] pub fn set_from(&mut self, vals: &[T]) where T: Copy {
        assert!(vals.len() == self.cols.len(), "set_from: length mismatch");
        for (dst, &v) in self.cols.iter_mut().zip(vals.iter()) { **dst = v; }
    }
}

// ============================================================================
// ------------------------------ Constructors --------------------------------
// ============================================================================

impl<T: Scalar> VectorList<T> {
    /// Create a new list with `n` vectors of dimensionality `dim`, default-initialized.
    /// Internally calls `Tensor::new(vec![dim, n])`.
    #[inline]
    pub fn new(dim: usize, n: usize) -> Self {
        assert!(dim > 0, "VectorList::new: dim must be > 0");
        Self { data: Tensor::new(vec![dim, n]) }
    }

    /// Wrap an existing dense tensor as a `VectorList`, asserting shape `[D, n]`.
    #[inline]
    pub fn from_tensor(t: Tensor<T>) -> Self {
        assert!(t.shape.len() == 2, "VectorList expects rank-2 tensor [D, n]");
        Self { data: t }
    }

    /// Number of vectors `n` (second dimension).
    #[inline]
    pub fn num_vectors(&self) -> usize {
        debug_assert!(self.data.shape.len() == 2);
        self.data.shape[1]
    }

    /// Dimensionality `D` (first dimension).
    #[inline]
    pub fn dim(&self) -> usize {
        debug_assert!(self.data.shape.len() == 2);
        self.data.shape[0]
    }

    /// Borrow the underlying tensor (shape `[D, n]`).
    #[inline] pub fn as_tensor(&self) -> &Tensor<T> { &self.data }

    /// Consume and return the underlying tensor (shape `[D, n]`).
    #[inline] pub fn into_tensor(self) -> Tensor<T> { self.data }
}

// ============================================================================
// ----------------------------- Accessors ------------------------------------
// ============================================================================

impl<T: Scalar> VectorList<T> {
    /// Immutable slice of dimension `k` (length `n`). Contiguous SoA chunk.
    #[inline]
    pub fn dim_slice(&self, k: usize) -> &[T] {
        let d = self.dim(); assert!(k < d, "dim_slice: k {} out of range (D={})", k, d);
        let n = self.num_vectors();
        let (start, end) = (k * n, (k + 1) * n);
        &self.data.data[start..end]
    }

    /// Mutable slice of dimension `k` (length `n`). Contiguous SoA chunk.
    #[inline]
    pub fn dim_slice_mut(&mut self, k: usize) -> &mut [T] {
        let d = self.dim(); assert!(k < d, "dim_slice_mut: k {} out of range (D={})", k, d);
        let n = self.num_vectors();
        let (start, end) = (k * n, (k + 1) * n);
        &mut self.data.data[start..end]
    }

    /// Immutable scalar access to component `(i,k)`.
    #[inline]
    pub fn get(&self, i: usize, k: usize) -> &T {
        let d = self.dim(); let n = self.num_vectors();
        assert!(k < d && i < n, "get: index out of range (i={}, k={}, D={}, n={})", i, k, d, n);
        self.data.get(&[k, i])
    }

    /// Mutable scalar access to component `(i,k)`.
    #[inline]
    pub fn get_mut(&mut self, i: usize, k: usize) -> &mut T {
        let d = self.dim(); let n = self.num_vectors();
        assert!(k < d && i < n, "get_mut: index out of range (i={}, k={}, D={}, n={})", i, k, d, n);
        self.data.get_mut(&[k, i])
    }

    /// Set component `(i,k)` to `val`.
    #[inline]
    pub fn set(&mut self, i: usize, k: usize, val: T) {
        let d = self.dim(); let n = self.num_vectors();
        assert!(k < d && i < n, "set: index out of range (i={}, k={}, D={}, n={})", i, k, d, n);
        self.data.set(&[k, i], val);
    }

    /// Return the `i`-th vector as a new `Vec<T>` of length `D` (copied).
    #[inline]
    pub fn get_vec(&self, i: usize) -> Vec<T> where T: Copy {
        let d = self.dim(); let n = self.num_vectors();
        assert!(i < n, "get_vec: i {} out of range (n={})", i, n);
        let mut out = Vec::with_capacity(d);
        for k in 0..d {
            let flat = k * n + i;
            out.push(*self.data.get_by_idx(flat));
        }
        out
    }

    /// Create a mutable proxy to the `i`-th vector exposing `D` disjoint `&mut T`.
    #[inline]
    pub fn get_vec_mut(&mut self, i: usize) -> VecRefMut<'_, T> {
        let d = self.dim(); let n = self.num_vectors();
        assert!(i < n, "get_vec_mut: i {} out of range (n={})", i, n);

        // SAFETY: indices (k*n + i) are all distinct for fixed `i`.
        let p = self.data.data.as_mut_ptr();
        let cols = (0..d).map(|k| unsafe { &mut *p.add(k * n + i) }).collect();
        VecRefMut { cols }
    }
}

// ============================================================================
// ---------------------------- Normalization ---------------------------------
// ============================================================================

impl<T> VectorList<T>
where
    T: Scalar + PartialOrd,
{
    /// In-place L2 normalization of each vector `i`: `v_i ← v_i / ||v_i||` if `||v_i|| > 0`.
    #[inline]
    pub fn normalize(&mut self) {
        let d = self.dim(); let n = self.num_vectors();
        if n == 0 || d == 0 { return; }

        // 1) norms: [n]
        let mut norms = Tensor::<T>::new(vec![n]);
        norms.data.par_iter_mut().enumerate().for_each(|(i, ni)| {
            let mut s = T::zero();
            for k in 0..d {
                let x = self.data.data[k * n + i];
                s = s + x * x;
            }
            *ni = <T as Scalar>::sqrt(s);
        });

        // 2) divide each SoA slice by norms
        for k in 0..d {
            let xs = self.dim_slice_mut(k);
            let ns = &norms.data;
            xs.par_iter_mut().zip(ns.par_iter()).for_each(|(x, &nrm)| {
                if nrm > T::zero() { *x = *x / nrm; }
            });
        }
    }

    /// Return `(norms, units)` where norms: `[n]`, units: `[D,n]` (zero maps to zero unit).
    #[inline]
    pub fn to_polar(&self) -> (Tensor<T>, VectorList<T>) {
        let d = self.dim(); let n = self.num_vectors();

        // 1) norms
        let mut norms = Tensor::<T>::new(vec![n]);
        norms.data.par_iter_mut().enumerate().for_each(|(i, ni)| {
            let mut s = T::zero();
            for k in 0..d {
                let flat = k * n + i;
                let x = *self.data.get_by_idx(flat);
                s = s + x * x;
            }
            *ni = <T as Scalar>::sqrt(s);
        });

        // 2) units
        let mut units = VectorList::<T>::new(d, n);
        for k in 0..d {
            let src = self.dim_slice(k);
            let dst = units.dim_slice_mut(k);
            dst.par_iter_mut().enumerate().for_each(|(i, u)| {
                let nrm = norms.data[i];
                let x = unsafe { *src.get_unchecked(i) };
                *u = if nrm > T::zero() { x / nrm } else { T::zero() };
            });
        }

        (norms, units)
    }
}

// ============================================================================
// ------------------------------ Arithmetics ---------------------------------
// ============================================================================

impl<T: Scalar> VectorList<T> {
    #[inline]
    fn assert_same_shape(&self, other: &Self) {
        assert!(self.data.shape.len() == 2 && other.data.shape.len() == 2);
        assert!(self.dim() == other.dim(), "D mismatch: {} vs {}", self.dim(), other.dim());
        assert!(self.num_vectors() == other.num_vectors(), "n mismatch: {} vs {}", self.num_vectors(), other.num_vectors());
    }
}

macro_rules! impl_binop_delegate {
    ($trait:ident, $method:ident) => {
        impl<T> core::ops::$trait for VectorList<T>
        where
            T: Scalar + core::ops::$trait<Output = T> + Send + Sync,
        {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                self.assert_same_shape(&rhs);
                let data = self.data.$method(rhs.data);
                VectorList { data }
            }
        }
    };
}

impl_binop_delegate!(Add, add);
impl_binop_delegate!(Sub, sub);
impl_binop_delegate!(Mul, mul);
impl_binop_delegate!(Div, div);
impl_binop_delegate!(BitAnd, bitand);

// ============================================================================
// ---------------------------- High Level Ops --------------------------------
// ============================================================================

impl<T: Scalar> VectorList<T> {
    /// Scale each vector `i` by `scales[i]`. Requires `scales.len() == n`.
    #[inline]
    pub fn scale_by_list(mut self, scales: &[T]) -> Self {
        let n = self.num_vectors();
        assert!(scales.len() == n, "scale_by_list: scales.len()={}, n={}", scales.len(), n);

        let d = self.dim();
        for k in 0..d {
            let slice = self.dim_slice_mut(k);
            slice.par_iter_mut().enumerate().for_each(|(i, x)| {
                *x = *x * unsafe { *scales.get_unchecked(i) };
            });
        }
        self
    }
}


// ============================================================================
// ---------------------------- Casting --------------------------------
// ============================================================================
impl<T: Scalar> VectorList<T> {
    /// Cast the underlying `[D, n]` tensor into a new `VectorList<U>`, consuming `self`.
    /// Delegates to `Tensor::cast_to::<U>()`.
    #[inline]
    pub fn cast_to<U: Scalar>(self) -> VectorList<U> {
        VectorList { data: self.data.cast_to::<U>() }
    }

    /// Cast into a new `VectorList<U>` without consuming `self` (clones first).
    /// Useful when you want to keep the original list.
    #[inline]
    pub fn cast_to_ref<U: Scalar>(&self) -> VectorList<U> {
        VectorList { data: self.data.cast_to::<U>() }
    }
}

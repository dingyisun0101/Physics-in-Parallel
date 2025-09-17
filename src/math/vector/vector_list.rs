// src/math_foundations/vector_list.rs
/*
    VectorList is a wrapper for tensor::dense::Tensor.
    It interprets tensor in a SoA layout [D, n] to represent an n-element list of D-dimensional vectors.
    It provides high level accesors to manipulate the vectors.
*/

use rayon::prelude::*;
use serde::Serialize;

use super::super::scalar::Scalar;
use super::super::tensor::dense::Tensor;


// ============================================================================
// --------------------------- Struct Def: SoA --------------------------------
// ============================================================================

/// Structure-of-Arrays list of n vectors of fixed dimensionality D,
#[derive(Debug, Clone, Serialize)]
pub struct VectorList<T: Scalar, const D: usize> {
    pub data: Tensor<T>, // shape = [D, n]
}

/// Small mutable view over one vector (row) in SoA.
pub struct VecRefMut<'a, T, const D: usize> {
    cols: [&'a mut T; D],
}
impl<'a, T: Scalar, const D: usize> VecRefMut<'a, T, D> {
    #[inline] pub fn get(&self, k: usize) -> &T { self.cols[k] }
    #[inline] pub fn get_mut(&mut self, k: usize) -> &mut T { self.cols[k] }
    #[inline] pub fn set_from(&mut self, vals: [T; D]) where T: Copy {
        for k in 0..D { *self.cols[k] = vals[k]; }
    }
}

impl<T: Scalar, const D: usize> VectorList<T, D> {
    /// Create with `n` vectors (all components default-initialized).
    #[inline]
    pub fn new(n: usize) -> Self {
        Self { data: Tensor::new(vec![D, n]) }
    }

    /// Wrap an existing SoA tensor (must be shape [D, n]).
    #[inline]
    pub fn from_tensor(t: Tensor<T>) -> Self {
        assert!(t.shape.len() == 2 && t.shape[0] == D, "VectorList expects shape [D, n]");
        Self { data: t }
    }

    /// Number of vectors `n` (derived from the Tensor shape).
    #[inline]
    pub fn num_vec(&self) -> usize {
        assert!(self.data.shape.len() == 2 && self.data.shape[0] == D, "shape must be [D, n]");
        self.data.shape[1]
    }

    /// Dimensionality (compile-time constant).
    #[inline]
    pub const fn dim(&self) -> usize { D }

    /// Borrow the underlying tensor (shape [D, n]).
    #[inline] pub fn as_tensor(&self) -> &Tensor<T> { &self.data }

    /// Consume into the underlying tensor (shape [D, n]).
    #[inline] pub fn into_tensor(self) -> Tensor<T> { self.data }
}





// ============================================================================
// ----------------------------- Accessors ------------------------------------
// ============================================================================
impl<T: Scalar, const D: usize> VectorList<T, D> {
    /// Immutable slice of dimension `k` (length = n). SoA-contiguous.
    #[inline]
    pub fn dim_slice(&self, k: usize) -> &[T] {
        assert!(k < D, "dim_slice: k {} out of range (D={})", k, D);
        let n = self.num_vec();
        let (start, end) = (k * n, (k + 1) * n);
        &self.data.data[start..end]
    }

    /// Mutable slice of dimension `k` (length = n). SoA-contiguous.
    #[inline]
    pub fn dim_slice_mut(&mut self, k: usize) -> &mut [T] {
        assert!(k < D, "dim_slice_mut: k {} out of range (D={})", k, D);
        let n = self.num_vec();
        let (start, end) = (k * n, (k + 1) * n);
        &mut self.data.data[start..end]
    }

    /// Scalar access (i, k) = i-th vector’s k-th component.
    #[inline]
    pub fn get(&self, i: usize, k: usize) -> &T {
        assert!(k < D, "get: k {} out of range (D={})", k, D);
        let n = self.num_vec();
        assert!(i < n, "get: i {} out of range (n={})", i, n);
        &self.data.data[k * n + i]
    }

    /// Mutable scalar access (i, k).
    #[inline]
    pub fn get_mut(&mut self, i: usize, k: usize) -> &mut T {
        assert!(k < D, "get_mut: k {} out of range (D={})", k, D);
        let n = self.num_vec();
        assert!(i < n, "get_mut: i {} out of range (n={})", i, n);
        &mut self.data.data[k * n + i]
    }

    /// Return i-th vector as a tiny copied array `[T; D]` (no heap).
    #[inline]
    pub fn get_vec(&self, i: usize) -> [T; D]
    where
        T: Copy,
    {
        let n = self.num_vec();
        assert!(i < n, "get_vec: i {} out of range (n={})", i, n);
        std::array::from_fn(|k| unsafe {
            // Safety: bounds checked above; k<n dims, SoA layout index = k*n + i
            *self.data.data.get_unchecked(k * n + i)
        })
    }

    /// Mutable proxy to i-th vector providing D disjoint `&mut T`.
    /// Uses `unsafe` internally to construct element-wise &mut refs at offsets `k*n + i`.
    #[inline]
    pub fn get_vec_mut(&mut self, i: usize) -> VecRefMut<'_, T, D> {
        let n = self.num_vec();
        assert!(i < n, "get_vec_mut: i {} out of range (n={})", i, n);
        // SAFETY: For fixed `i`, the D indices `k*n + i` are all distinct for k=0..D-1.
        let p = self.data.data.as_mut_ptr();
        let cols = std::array::from_fn(|k| unsafe { &mut *p.add(k * n + i) });
        VecRefMut { cols }
    }
}


// ============================================================================
// ---------------------------- Normalization ---------------------------------
// ============================================================================

impl<T, const D: usize> VectorList<T, D>
where
    T: Scalar + PartialOrd
{
    // In-place L2 normalization
    #[inline]
    pub fn normalize(&mut self) {
        let n = self.num_vec();

        for i in 0..n {
            // 1) compute norm of row i
            let mut s = T::zero();
            for k in 0..D {
                let idx = k * n + i;                // SoA index (k, i)
                let x   = self.data.data[idx];
                s = s + x * x;
            }
            let nrm = <T as Scalar>::sqrt(s);

            // 2) scale row i if nonzero
            if nrm > T::zero() {
                for k in 0..D {
                    let idx = k * n + i;
                    // write back normalized component
                    self.data.data[idx] = self.data.data[idx] / nrm;
                }
            }
        }
    }

    /// Compute per-vector L2 norms and unit vectors.
    /// Allocates only the returned outputs:
    /// - `norms`: Tensor<T> with shape [n]
    /// - `units`: VectorList<T, D> with shape [D, n]
    #[inline]
    pub fn to_polar(&self) -> (Tensor<T>, VectorList<T, D>) {
        let n = self.num_vec();

        // -------- 1) norms: [n] --------
        let mut norms = Tensor::<T>::new(vec![n]);
        norms
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, ni)| {
                let mut s = T::zero();
                // sum over dims
                for k in 0..D {
                    let x = unsafe { *self.data.data.get_unchecked(k * n + i) };
                    s = s + x * x;
                }
                *ni = <T as Scalar>::sqrt(s);
            });

        // -------- 2) unit vectors: [D, n] --------
        let mut units = VectorList::<T, D>::new(n);
        for k in 0..D {
            // source and destination column slices (length n)
            let src_start = k * n;
            let src = &self.data.data[src_start .. src_start + n];
            let dst = &mut units.data.data[src_start .. src_start + n];

            dst.par_iter_mut()
               .enumerate()
               .for_each(|(i, u)| {
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

// Arithmetic ops: delegate to `Tensor` after shape checks.
// These consume `self` and `rhs` (matching your `Tensor` ops signatures).

impl<T, const D: usize> VectorList<T, D>
where
    T: Scalar,
{
    #[inline]
    fn assert_same_shape(&self, other: &Self) {
        // Both must be [D, n] with the same n.
        assert!(
            self.data.shape.len() == 2 && self.data.shape[0] == D,
            "lhs shape must be [D, n], got {:?}",
            self.data.shape
        );
        assert!(
            other.data.shape.len() == 2 && other.data.shape[0] == D,
            "rhs shape must be [D, n], got {:?}",
            other.data.shape
        );
        assert!(
            self.num_vec() == other.num_vec(),
            "VectorList length mismatch: {} vs {}",
            self.num_vec(),
            other.num_vec()
        );
    }
}

macro_rules! impl_binop_delegate {
    ($trait:ident, $method:ident) => {
        impl<T, const D: usize> core::ops::$trait for VectorList<T, D>
        where
            T: Scalar + core::ops::$trait<Output = T>,
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

impl<T: Scalar, const D: usize> VectorList<T, D> {
    /// Scale each i-th vector by `scales[i]`. Length of `scales` must be `n`.
    #[inline]
    pub fn scale_by_list(mut self, scales: &Vec<T>) -> Self {
        let n = self.num_vec();
        assert!(
            scales.len() == n,
            "scale_by_list: scales.len() = {}, expected n = {}",
            scales.len(),
            n
        );

        // For each dimension k, multiply the column slice [k*n .. (k+1)*n]
        // elementwise by scales[i].
        for k in 0..D {
            let start = k * n;
            let dst = &mut self.data.data[start .. start + n];

            dst.par_iter_mut()
               .enumerate()
               .for_each(|(i, x)| {
                   *x = *x * unsafe { *scales.get_unchecked(i) };
               });
        }
        self
    }
}

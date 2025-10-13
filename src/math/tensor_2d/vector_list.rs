// src/math_foundations/tensor_2d/vector_list.rs
/*!
A **SoA (structure-of-arrays)** container for many fixed-length vectors,
backed by a generic 2D `Matrix<T, B>` with shape `[dim, n]`.

- **Rows**   = coordinates / axes (dimension `dim`)
- **Columns**= vectors (count `n`)
- **Invariant**: each **column is a vector** of length `dim`.

This wrapper provides ergonomic row/column views, bulk set/fill, scaling,
and simple arithmetic forwarding to the `Matrix` backend.
*/
use core::marker::PhantomData;
use rayon::prelude::*;

use crate::math::{
    scalar::Scalar,
    tensor::{
        dense::Tensor,
        tensor_trait::TensorTrait,
    },
    tensor_2d::matrix::{Matrix, RowRef, RowRefMut, ColRef, ColRefMut}
};

// ============================================================================
// ------------------------------- Core Structs -------------------------------
// ============================================================================
#[derive(Debug, Clone)]
pub struct VectorList<T: Scalar, B: TensorTrait<T> = Tensor<T>> {
    pub matrix: Matrix<T, B>, // rows = dim (D), cols = n
    _pd: PhantomData<T>,
}

// ============================================================================
// --------------------------------- Basic I/O --------------------------------
// ============================================================================

impl<T: Scalar, B: TensorTrait<T>> VectorList<T, B> {
    /// New `[dim, n]` SoA container.
    #[inline]
    pub fn empty(dim: usize, n: usize) -> Self {
        assert!(dim > 0 && n > 0, "VectorList::new: shape must be nonzero");
        Self {
            matrix: Matrix::<T, B>::empty(dim, n),
            _pd: PhantomData,
        }
    }

    /// Wrap an existing matrix (must be rank-2).
    #[inline]
    pub fn from_matrix(m: Matrix<T, B>) -> Self {
        // trust Matrix invariants
        Self { matrix: m, _pd: PhantomData }
    }

    /// Extract the backing `Matrix`.
    #[inline]
    pub fn into_matrix(self) -> Matrix<T, B> { self.matrix }

    // ----------------------- shape helpers -----------------------
    #[inline] pub fn dim(&self) -> usize { self.matrix.shape()[0] }
    #[inline] pub fn num_vectors(&self) -> usize { self.matrix.shape()[1] }
    #[inline] pub fn shape(&self) -> &[usize] { self.matrix.shape() }

    // ------------------------ Type helpers ------------------------
    #[inline]
    pub fn cast_to<U>(&self) -> VectorList<U, B::Repr<U>>
    where
        U: Scalar + Send + Sync,
    {
        let data_u: Matrix<U, B::Repr<U>> = self.matrix.cast_to::<U>();
        VectorList { matrix: data_u, _pd: PhantomData }
    }

    #[inline]
    pub fn print(&self) {
        self.matrix.print();
    }
}

// ============================================================================
// --------------------------------- Accessors --------------------------------
// ============================================================================

impl<T: Scalar, B: TensorTrait<T>> VectorList<T, B> {
    // --------------------- element accessors ---------------------
    /// Get by value at vector `i` (column), axis `k` (row).
    #[inline]
    pub fn get(&self, i: isize, k: isize) -> T where T: Copy {
        self.matrix.get(k, i)
    }

    /// Set at vector `i` (column), axis `k` (row).
    #[inline]
    pub fn set(&mut self, i: isize, k: isize, val: T) {
        self.matrix.set(k, i, val);
    }

    /// Mutable ref to entry `(i, k)` (vector `i`, axis `k`).
    #[inline]
    pub fn get_mut(&mut self, i: isize, k: isize) -> &mut T {
        self.matrix.get_mut(k, i)
    }

    // ------------------------- views -----------------------------
    /// Axis view (row) for dimension `k`.
    #[inline] pub fn get_axis(&self, k: isize) -> RowRef<'_, T, B> { self.matrix.row(k) }
    #[inline] pub fn get_axis_mut(&mut self, k: isize) -> RowRefMut<'_, T, B> { self.matrix.row_mut(k) }

    /// Vector view (column) for vector `i`.
    #[inline] pub fn get_vector(&self, i: isize) -> ColRef<'_, T, B> { self.matrix.col(i) }
    #[inline] pub fn get_vector_mut(&mut self, i: isize) -> ColRefMut<'_, T, B> { self.matrix.col_mut(i) }
}

// ============================================================================
// ---------------------------- Bulk Operations -------------------------------
// ============================================================================

/// Scaling
impl<T, B> VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
{
    /// Scale each **vector (column)** by the corresponding factor in `scales`.
    ///
    /// - `scales.len()` must equal `self.num_vectors()`.
    /// - Parallelized across **columns** to build scaled values, then committed.
    #[inline]
    pub fn scale_vectors_by_list(&mut self, scales: &[T]) {
        let d = self.dim();
        let n = self.num_vectors();
        assert!(scales.len() == n, "scale_vectors_by_list: len mismatch");

        // Phase 1 (parallel): compute scaled columns (each of length d)
        let scaled_cols: Vec<Vec<T>> = (0..n)
            .into_par_iter()
            .map(|k| {
                let s = scales[k];
                let mut out = Vec::with_capacity(d);
                for row in 0..d {
                    let x = self.matrix.get(row as isize, k as isize);
                    out.push(x * s); // scale column k by scales[k]
                }
                out
            })
            .collect();

        // Phase 2 (sequential): commit back column by column.
        for (k, col_vals) in scaled_cols.into_iter().enumerate() {
            // len(col_vals) == d, matches Matrix::col_mut(k).set_from_slice
            self.matrix.col_mut(k as isize).set_from_slice(&col_vals);
        }
    }

    /// Set an entire **vector** (column) `i` from slice `vals` of length `dim()`.
    #[inline]
    pub fn set_vector_from_slice(&mut self, i: isize, vals: &[T])
    where
        T: Copy,
    {
        assert!(vals.len() == self.dim(), "set_vector_from_slice: len mismatch");
        self.matrix.col_mut(i).set_from_slice(vals);
    }

    /// Set an entire **axis** (row) `k` from slice `vals` of length `num_vectors()`.
    #[inline]
    pub fn set_axis_from_slice(&mut self, k: isize, vals: &[T])
    where
        T: Copy,
    {
        assert!(vals.len() == self.num_vectors(), "set_axis_from_slice: len mismatch");
        self.matrix.row_mut(k).set_from_slice(vals);
    }
}

// ============================================================================
// ------------------------------ Math Utils ----------------------------------
// ============================================================================

/// Norms + normalization
impl<T, B> VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T> + Send + Sync,
{
    /// Return the L2 norms of each **vector (column)** as `Vec<T>`.
    /// Parallelized across columns with Rayon.
    #[inline]
    pub fn get_norms(&self) -> Vec<T> {
        let d = self.dim();
        let n = self.num_vectors();

        (0..n).into_par_iter()
            .map(|col| {
                let mut s = T::zero();
                for row in 0..d {
                    let x = self.matrix.get(row as isize, col as isize);
                    s = s + x * x;
                }
                <T>::sqrt(s)
            })
            .collect()
    }

    /// In-place column-wise normalization to unit L2 norm.
    #[inline]
    pub fn normalize(&mut self) {
        let norms = self.get_norms();
        let scales: Vec<T> = norms.par_iter()
            .map(|&n| if n == T::zero() { T::one() } else { T::one() / n })
            .collect();
        self.scale_vectors_by_list(&scales);
    }

    /// Return `(norms, unit_vectors)` where `unit_vectors[:, j] = v[:, j] / ||v_j||`.
    #[inline]
    pub fn to_polar(&self) -> (Vec<T>, VectorList<T, B>)
    where
        T: Scalar,
        B: TensorTrait<T>,
    {
        let norms = self.get_norms();
        let mut units = self.clone();

        // scale each column by 1 / ||v_j|| (safe when norm == 0)
        let scales: Vec<T> = norms.iter()
            .map(|&n| if n == T::zero() { T::zero() } else { T::one() / n })
            .collect();

        units.scale_vectors_by_list(&scales);
        (norms, units)
    }
}



// ============================================================================
// --------------------------- Arithmetic Ops ---------------------------------
// ============================================================================

// ---------------------------- Macro helpers ---------------------------------

// &VectorList ⊕ &VectorList -> VectorList  (directly on backend tensors)
macro_rules! impl_vl_ref_binop_backend {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T, B> core::ops::$trait<&'a VectorList<T, B>> for &'a VectorList<T, B>
        where
            T: Scalar,
            B: TensorTrait<T>,
            for<'b> &'b B: core::ops::$trait<&'b B, Output = B>,
        {
            type Output = VectorList<T, B>;
            #[inline]
            fn $method(self, rhs: &'a VectorList<T, B>) -> Self::Output {
                let tensor: B = &self.matrix.tensor $op &rhs.matrix.tensor;
                let matrix = Matrix::from_tensor(tensor);
                VectorList { matrix, _pd: core::marker::PhantomData }
            }
        }
    };
}

// VectorList op= VectorList (compute via backend zip)
macro_rules! impl_vl_ref_assign_backend {
    ($assign_trait:ident, $assign_method:ident, $elem_trait:ident, $op:tt) => {
        impl<T, B> core::ops::$assign_trait<&VectorList<T, B>> for VectorList<T, B>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$elem_trait<Output = T>,
            B: TensorTrait<T>,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: &VectorList<T, B>) {
                self.matrix.tensor
                    .par_zip_with_inplace(&rhs.matrix.tensor, |x, y| x $op y);
            }
        }
    };
}

// &VectorList ⊕ scalar -> VectorList (directly on backend tensors)
macro_rules! impl_vl_ref_scalar_binop_backend {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T, B> core::ops::$trait<T> for &'a VectorList<T, B>
        where
            T: Scalar,
            B: TensorTrait<T>,
            for<'b> &'b B: core::ops::$trait<T, Output = B>,
        {
            type Output = VectorList<T, B>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                let tensor: B = &self.matrix.tensor $op rhs;
                let matrix = Matrix::from_tensor(tensor);
                VectorList { matrix, _pd: core::marker::PhantomData }
            }
        }
    };
}

// VectorList scalar-op= (delegate to backend scalar-assign)
macro_rules! impl_vl_scalar_assign_backend {
    ($assign_trait:ident, $assign_method:ident) => {
        impl<T, B> core::ops::$assign_trait<T> for VectorList<T, B>
        where
            T: Scalar,
            B: TensorTrait<T> + core::ops::$assign_trait<T>,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: T) {
                self.matrix.tensor.$assign_method(rhs);
            }
        }
    };
}

// ------------------------------- Invocations --------------------------------

// &A ⊕ &B
impl_vl_ref_binop_backend!(Add, add, +);
impl_vl_ref_binop_backend!(Sub, sub, -);
impl_vl_ref_binop_backend!(Mul, mul, *);
impl_vl_ref_binop_backend!(Div, div, /);

// A op= B   (uses elementwise trait bound for the closure type)
impl_vl_ref_assign_backend!(AddAssign, add_assign, Add, +);
impl_vl_ref_assign_backend!(SubAssign, sub_assign, Sub, -);
impl_vl_ref_assign_backend!(MulAssign, mul_assign, Mul, *);
impl_vl_ref_assign_backend!(DivAssign, div_assign, Div, /);

// &A ⊕ scalar
impl_vl_ref_scalar_binop_backend!(Add, add, +);
impl_vl_ref_scalar_binop_backend!(Sub, sub, -);
impl_vl_ref_scalar_binop_backend!(Mul, mul, *);
impl_vl_ref_scalar_binop_backend!(Div, div, /);

// A scalar-op=
impl_vl_scalar_assign_backend!(AddAssign, add_assign);
impl_vl_scalar_assign_backend!(SubAssign, sub_assign);
impl_vl_scalar_assign_backend!(MulAssign, mul_assign);
impl_vl_scalar_assign_backend!(DivAssign, div_assign);




/*!
A **SoA (structure-of-arrays)** container for many fixed-length vectors,
backed by a dense `Matrix<T>` with shape `[dim, n]`.

- **Rows**    = coordinates / axes (dimension `dim`)
- **Columns** = vectors (count `n`)
- **Invariant**: each **column is a vector** of length `dim`.

This chooses **column-major** so `get_vector(i)` is zero-copy.
*/
use rayon::prelude::*;

use crate::math::{
    scalar::Scalar,
    tensor::{
        dense::Tensor as DenseTensor,
        tensor_trait::TensorTrait,
    },
    tensor_2d::matrix::{
        matrix_trait::MatrixTrait,
        dense::{Matrix, Major, MatrixSlice, MatrixSliceMut},
    }
};

#[derive(Debug, Clone)]
pub struct VectorList<T: Scalar> {
    /// Backing matrix with shape `[dim, n]` (column-major for zero-copy vectors).
    pub matrix: Matrix<T>,
}

impl<T: Scalar> VectorList<T> {
    /// New `[dim, n]` SoA container filled with **zeros** (column-major).
    #[inline]
    pub fn empty(dim: usize, n: usize) -> Self {
        assert!(dim > 0 && n > 0, "VectorList::empty: shape must be nonzero");
        let matrix = Matrix::<T>::empty(dim, n, Major::Col);
        Self { matrix }
    }

    /// Wrap an existing dense `Matrix<T>`. Caller should ensure `major == Col`.
    #[inline]
    pub fn from_matrix(m: Matrix<T>) -> Self {
        debug_assert!(matches!(m.major(), Major::Col), "VectorList expects column-major matrix");
        Self { matrix: m }
    }

    /// Extract the backing `Matrix<T>`.
    #[inline]
    pub fn into_matrix(self) -> Matrix<T> { self.matrix }

    /// Borrow the underlying dense tensor (physical layout depends on major).
    #[inline]
    pub fn as_tensor(&self) -> &DenseTensor<T> { self.matrix.backend() }

    /// Mutably borrow the underlying dense tensor.
    #[inline]
    pub fn as_tensor_mut(&mut self) -> &mut DenseTensor<T> { self.matrix.backend_mut() }

    // ----------------------- shape helpers -----------------------
    #[inline] pub fn dim(&self) -> usize { self.matrix.rows() }
    #[inline] pub fn num_vectors(&self) -> usize { self.matrix.cols() }
    #[inline] pub fn shape(&self) -> [usize; 2] { [self.dim(), self.num_vectors()] }

    // ------------------------ I/O helpers ------------------------
    #[inline]
    pub fn print(&self) { self.matrix.print(); }

    /// Cast the scalar type of all vectors (keeps `dim`, `n`, and column-major layout).
    #[inline]
    pub fn cast_to<U>(&self) -> VectorList<U>
    where
        U: Scalar,
    {
        let m_u = self.matrix.cast_to::<U>();
        VectorList { matrix: m_u }
    }

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

    // ------------------------- views -----------------------------

    /// Zero-copy **vector** (column) view (length = `dim()`).
    #[inline]
    pub fn get_vector(&self, i: isize) -> MatrixSlice<'_, T>
    where
        T: Copy,
    {
        self.matrix.get_col_ref(i)
    }

    /// Mutable **vector** (column) view. Zero-copy in column-major,
    /// or guarded (commit-on-drop) if ever used with row-major.
    #[inline]
    pub fn get_vector_mut(&mut self, i: isize) -> MatrixSliceMut<'_, T>
    where
        T: Copy,
    {
        self.matrix.get_col_ref_mut(i)
    }

    /// Copy **axis** (row) `k` into a new `Vec<T>` (length = `num_vectors()`).
    #[inline]
    pub fn axis_to_vec(&self, k: isize) -> Vec<T>
    where
        T: Copy,
    {
        // Use the immutable row view; it may be cached when col-major.
        let r = self.matrix.get_row_ref(k);
        let mut v = vec![T::zero(); r.len()];
        v.copy_from_slice(&r);
        v
    }

    /// Copy **vector** (column) `i` into a new `Vec<T>` (length = `dim()`).
    #[inline]
    pub fn vector_to_vec(&self, i: isize) -> Vec<T>
    where
        T: Copy,
    {
        let c = self.get_vector(i);
        let mut v = vec![T::zero(); c.len()];
        v.copy_from_slice(&c);
        v
    }
}

// ============================================================================
// ---------------------------- Bulk Operations -------------------------------
// ============================================================================

impl<T> VectorList<T>
where
    T: Scalar + Copy + Send + Sync,
{
    /// Scale each **vector (column)** by the corresponding factor in `scales`.
    ///
    /// In column-major, each column is contiguous so this mostly writes linearly.
    #[inline]
    pub fn scale_vectors_by_list(&mut self, scales: &[T]) {
        let n = self.num_vectors();
        assert!(scales.len() == n, "scale_vectors_by_list: len mismatch");

        for col in 0..n {
            let s = scales[col];
            match self.matrix.get_col_ref_mut(col as isize) {
                MatrixSliceMut::Borrowed(dst) => {
                    for x in dst {
                        *x = *x * s;
                    }
                }
                MatrixSliceMut::Guard(_) => {
                    unreachable!("VectorList is column-major: mutable vector view must be contiguous")
                }
            }
        }
    }

    /// Set an entire **vector** (column) `i` from slice `vals` (length = `dim()`).
    #[inline]
    pub fn set_vector_from_slice(&mut self, i: isize, vals: &[T]) {
        assert_eq!(vals.len(), self.dim(), "set_vector_from_slice: len mismatch");
        match self.matrix.get_col_ref_mut(i) {
            MatrixSliceMut::Borrowed(dst) => dst.copy_from_slice(vals),
            MatrixSliceMut::Guard(mut g)   => g.copy_from_slice(vals), // commit on Drop
        }
    }

    /// Set an entire **axis** (row) `k` from slice `vals` (length = `num_vectors()`).
    #[inline]
    pub fn set_axis_from_slice(&mut self, k: isize, vals: &[T]) {
        assert_eq!(vals.len(), self.num_vectors(), "set_axis_from_slice: len mismatch");
        match self.matrix.get_row_ref_mut(k) {
            MatrixSliceMut::Borrowed(dst) => dst.copy_from_slice(vals),
            MatrixSliceMut::Guard(mut g)   => g.copy_from_slice(vals), // commit on Drop
        }
    }

    /// Fill all entries with `val`. (Contiguous in the physical tensor.)
    #[inline]
    pub fn fill(&mut self, val: T) {
        self.matrix.backend_mut().par_fill(val);
    }

    /// Return the L2 norms of each **vector (column)** as `Vec<T>`.
    #[inline]
    pub fn get_norms(&self) -> Vec<T> {
        let d = self.dim();
        let n = self.num_vectors();
        (0..n).into_par_iter()
            .map(|col| {
                let mut s = T::zero();
                let c = self.matrix.get_col_ref(col as isize);
                for i in 0..d {
                    let x = c[i];
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
    pub fn to_polar(&self) -> (Vec<T>, VectorList<T>) {
        let norms = self.get_norms();
        let mut units = self.clone();
        let scales: Vec<T> = norms.iter()
            .map(|&n| if n == T::zero() { T::zero() } else { T::one() / n })
            .collect();
        units.scale_vectors_by_list(&scales);
        (norms, units)
    }
}



// ============================================================================
// ----------------------------- Arithmetic Ops -------------------------------
// ============================================================================

// ------------------------ &VectorList ⊕ &VectorList -> VectorList -----------
macro_rules! impl_vl_ref_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> core::ops::$trait<&'a VectorList<T>> for &'a VectorList<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<Output = T>,
        {
            type Output = VectorList<T>;
            #[inline]
            fn $method(self, rhs: &'a VectorList<T>) -> Self::Output {
                let matrix = &self.matrix $op &rhs.matrix; // delegates to Matrix (tensor-backed)
                VectorList { matrix }
            }
        }
    };
}
impl_vl_ref_binop!(Add, add, +);
impl_vl_ref_binop!(Sub, sub, -);
impl_vl_ref_binop!(Mul, mul, *);
impl_vl_ref_binop!(Div, div, /);

// ------------------------ VectorList ⊕= &VectorList (in-place) --------------
macro_rules! impl_vl_ref_assign {
    ($trait:ident, $method:ident) => {
        impl<'a, T> core::ops::$trait<&'a VectorList<T>> for VectorList<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<T>,
        {
            #[inline]
            fn $method(&mut self, rhs: &'a VectorList<T>) {
                self.matrix.$method(&rhs.matrix);
            }
        }
    };
}
impl_vl_ref_assign!(AddAssign, add_assign);
impl_vl_ref_assign!(SubAssign, sub_assign);
impl_vl_ref_assign!(MulAssign, mul_assign);
impl_vl_ref_assign!(DivAssign, div_assign);

// ------------------------ &VectorList ⊕ scalar -> VectorList ----------------
macro_rules! impl_vl_ref_scalar_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> core::ops::$trait<T> for &'a VectorList<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<Output = T>,
        {
            type Output = VectorList<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                let matrix = &self.matrix $op rhs; // matrix-scalar delegates to tensor
                VectorList { matrix }
            }
        }
    };
}
impl_vl_ref_scalar_binop!(Add, add, +);
impl_vl_ref_scalar_binop!(Sub, sub, -);
impl_vl_ref_scalar_binop!(Mul, mul, *);
impl_vl_ref_scalar_binop!(Div, div, /);

// ------------------------ VectorList ⊕= scalar (in-place) -------------------
macro_rules! impl_vl_scalar_assign {
    ($trait:ident, $method:ident) => {
        impl<T> core::ops::$trait<T> for VectorList<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<T>,
        {
            #[inline]
            fn $method(&mut self, rhs: T) {
                self.matrix.$method(rhs);
            }
        }
    };
}
impl_vl_scalar_assign!(AddAssign, add_assign);
impl_vl_scalar_assign!(SubAssign, sub_assign);
impl_vl_scalar_assign!(MulAssign, mul_assign);
impl_vl_scalar_assign!(DivAssign, div_assign);
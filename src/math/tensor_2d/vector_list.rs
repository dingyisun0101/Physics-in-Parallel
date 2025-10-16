// src/math_foundations/tensor_2d/vector_list.rs
/*!
A **SoA (structure-of-arrays)** container for many fixed-length vectors,
backed by a generic 2D `Matrix<T>` with shape `[dim, n]`.

- **Rows**    = coordinates / axes (dimension `dim`)
- **Columns** = vectors (count `n`)
- **Invariant**: each **column is a vector** of length `dim`.

This wrapper provides ergonomic bulk set/fill/scaling and simple arithmetic
forwarding to the `Matrix` backend. It uses the **type-erased** `Matrix<T>`
API (dense or sparse selected at construction).
*/
use rayon::prelude::*;

use crate::math::{
    scalar::Scalar,
    tensor::dense::Tensor,
    tensor_2d::matrix::prelude::*,
};

// ============================================================================
// ------------------------------- Core Structs -------------------------------
// ============================================================================

#[derive(Debug, Clone)]
pub struct VectorList<T: Scalar> {
    /// Backing matrix with shape `[dim, n]` (type-erased dense/sparse).
    pub matrix: Matrix<T>,
}

// ============================================================================
// --------------------------------- Basic I/O --------------------------------
// ============================================================================

impl<T: Scalar> VectorList<T> {
    /// New `[dim, n]` SoA container filled with **zeros**.
    /// Choose backend at construction.
    #[inline]
    pub fn empty(dim: usize, n: usize, kind: BackendKind) -> Self {
        assert!(dim > 0 && n > 0, "VectorList::empty: shape must be nonzero");
        let zeros = vec![T::zero(); dim * n];
        let matrix = build_matrix(dim, n, &zeros, kind);
        Self { matrix }
    }

    /// Wrap an existing `Matrix<T>` (must be rank-2 with nonzero shape).
    #[inline]
    pub fn from_matrix(m: Matrix<T>) -> Self {
        // Matrix<T> already enforces rank-2 invariants internally.
        Self { matrix: m }
    }

    /// Extract the backing `Matrix<T>`.
    #[inline]
    pub fn into_matrix(self) -> Matrix<T> { self.matrix }

    #[inline]
    pub fn as_tensor<'a>(&'a self) -> &'a Tensor<T> {
        match &self.matrix {
            Matrix::Dense(m) => &m.tensor,
            _ => unreachable!("VectorList<T> is constructed only with a dense Matrix backend"),
        }
    }

    /// Mutably borrow the **dense** backend tensor.
    #[inline]
    pub fn as_tensor_mut<'a>(&'a mut self) -> &'a mut Tensor<T> {
        match &mut self.matrix {
            Matrix::Dense(m) => &mut m.tensor,
            _ => unreachable!("VectorList<T> is constructed only with a dense Matrix backend"),
        }
    }
    // ----------------------- shape helpers -----------------------
    #[inline] pub fn dim(&self) -> usize { self.matrix.rows() }
    #[inline] pub fn num_vectors(&self) -> usize { self.matrix.cols() }
    /// Returns `[dim, n]`.
    #[inline] pub fn shape(&self) -> [usize; 2] { [self.dim(), self.num_vectors()] }

    // ------------------------ Type helpers ------------------------
    #[inline]
    pub fn cast_to<U>(&self) -> VectorList<U>
    where
        U: Scalar,
    {
        let m_u: Matrix<U> = self.matrix.cast_to::<U>();
        VectorList { matrix: m_u }
    }

    #[inline]
    pub fn print(&self) { self.matrix.print(); }
}

// ============================================================================
// --------------------------------- Accessors --------------------------------
// ============================================================================

impl<T: Scalar> VectorList<T> {
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

    // ------------------------- “views” as copies -----------------
    /// Copy **axis** (row) `k` into a new `Vec<T>` (length = `num_vectors()`).
    #[inline]
    pub fn axis_to_vec(&self, k: isize) -> Vec<T> where T: Copy {
        self.matrix.row_to_vec(k)
    }

    /// Copy **vector** (column) `i` into a new `Vec<T>` (length = `dim()`).
    #[inline]
    pub fn vector_to_vec(&self, i: isize) -> Vec<T> where T: Copy {
        self.matrix.col_to_vec(i)
    }
}

// ============================================================================
// ---------------------------- Bulk Operations -------------------------------
// ============================================================================

impl<T> VectorList<T>
where
    T: Scalar,
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
            .map(|col| {
                let s = scales[col];
                let mut out = Vec::with_capacity(d);
                for row in 0..d {
                    let x = self.matrix.get(row as isize, col as isize);
                    out.push(x * s); // scale column col by scales[col]
                }
                out
            })
            .collect();

        // Phase 2 (sequential): commit back column by column.
        for (col, col_vals) in scaled_cols.into_iter().enumerate() {
            // len(col_vals) == d
            self.matrix.set_col_from_slice(col as isize, &col_vals);
        }
    }

    /// Set an entire **vector** (column) `i` from slice `vals` of length `dim()`.
    #[inline]
    pub fn set_vector_from_slice(&mut self, i: isize, vals: &[T])
    where
        T: Copy,
    {
        assert!(vals.len() == self.dim(), "set_vector_from_slice: len mismatch");
        self.matrix.set_col_from_slice(i, vals);
    }

    /// Set an entire **axis** (row) `k` from slice `vals` of length `num_vectors()`.
    #[inline]
    pub fn set_axis_from_slice(&mut self, k: isize, vals: &[T])
    where
        T: Copy,
    {
        assert!(vals.len() == self.num_vectors(), "set_axis_from_slice: len mismatch");
        self.matrix.set_row_from_slice(k, vals);
    }

    /// Fill all stored entries using backend semantics
    /// (dense: all elements, sparse: stored nonzeros).
    #[inline]
    pub fn fill(&mut self, val: T)
    where
        T: Copy + Send + Sync,
    {
        self.matrix.fill(val);
    }
}

// ============================================================================
// ------------------------------ Math Utils ----------------------------------
// ============================================================================

impl<T> VectorList<T>
where
    T: Scalar + Send + Sync,
{
    /// Return the L2 norms of each **vector (column)** as `Vec<T>`.
    /// Parallelized across columns with Rayon.
    #[inline]
    pub fn get_norms(&self) -> Vec<T> {
        let d = self.dim();
        let n = self.num_vectors();

        (0..n)
            .into_par_iter()
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
        let scales: Vec<T> = norms
            .par_iter()
            .map(|&n| if n == T::zero() { T::one() } else { T::one() / n })
            .collect();
        self.scale_vectors_by_list(&scales);
    }

    /// Return `(norms, unit_vectors)` where `unit_vectors[:, j] = v[:, j] / ||v_j||`.
    #[inline]
    pub fn to_polar(&self) -> (Vec<T>, VectorList<T>)
    where
        T: Copy,
    {
        let norms = self.get_norms();
        let mut units = self.clone();

        // scale each column by 1 / ||v_j|| (safe when norm == 0)
        let scales: Vec<T> = norms
            .iter()
            .map(|&n| if n == T::zero() { T::zero() } else { T::one() / n })
            .collect();

        units.scale_vectors_by_list(&scales);
        (norms, units)
    }
}

// ============================================================================
// --------------------------- Arithmetic Ops ---------------------------------
// ============================================================================

// &VectorList ⊕ &VectorList -> VectorList  (via Matrix enum ops)
macro_rules! impl_vl_ref_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> core::ops::$trait<&'a VectorList<T>> for &'a VectorList<T>
        where
            T: Scalar,
        {
            type Output = VectorList<T>;
            #[inline]
            fn $method(self, rhs: &'a VectorList<T>) -> Self::Output {
                let matrix = &self.matrix $op &rhs.matrix; // Matrix<T> (enum) arithmetic
                VectorList { matrix }
            }
        }
    };
}

// VectorList op= VectorList (delegate to Matrix enum op=)
macro_rules! impl_vl_ref_assign {
    ($assign_trait:ident, $assign_method:ident, $elem_trait:ident, $op:tt) => {
        impl<T> core::ops::$assign_trait<&VectorList<T>> for VectorList<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$elem_trait<Output = T>,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: &VectorList<T>) {
                self.matrix.$assign_method(&rhs.matrix);
            }
        }
    };
}

// &VectorList ⊕ scalar -> VectorList (via Matrix enum ops)
macro_rules! impl_vl_ref_scalar_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> core::ops::$trait<T> for &'a VectorList<T>
        where
            T: Scalar,
        {
            type Output = VectorList<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                let matrix = &self.matrix $op rhs;
                VectorList { matrix }
            }
        }
    };
}

// VectorList scalar-op= (delegate to Matrix enum)
macro_rules! impl_vl_scalar_assign {
    ($assign_trait:ident, $assign_method:ident) => {
        impl<T> core::ops::$assign_trait<T> for VectorList<T>
        where
            T: Scalar,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: T) {
                self.matrix.$assign_method(rhs);
            }
        }
    };
}

// ------------------------------- Invocations --------------------------------

// &A ⊕ &B
impl_vl_ref_binop!(Add, add, +);
impl_vl_ref_binop!(Sub, sub, -);
impl_vl_ref_binop!(Mul, mul, *);
impl_vl_ref_binop!(Div, div, /);

// A op= B
impl_vl_ref_assign!(AddAssign, add_assign, Add, +=);
impl_vl_ref_assign!(SubAssign, sub_assign, Sub, -=);
impl_vl_ref_assign!(MulAssign, mul_assign, Mul, *=);
impl_vl_ref_assign!(DivAssign, div_assign, Div, /=);

// &A ⊕ scalar
impl_vl_ref_scalar_binop!(Add, add, +);
impl_vl_ref_scalar_binop!(Sub, sub, -);
impl_vl_ref_scalar_binop!(Mul, mul, *);
impl_vl_ref_scalar_binop!(Div, div, /);

// A scalar-op=
impl_vl_scalar_assign!(AddAssign, add_assign);
impl_vl_scalar_assign!(SubAssign, sub_assign);
impl_vl_scalar_assign!(MulAssign, mul_assign);
impl_vl_scalar_assign!(DivAssign, div_assign);


// ============================================================================
// --------------------------------- Prelude ----------------------------------
// ============================================================================

/// Bring this into scope with:
/// ```ignore
/// 
/// ```
pub mod prelude {
    pub use crate::math::tensor_2d::matrix::prelude::*;
    pub use super::VectorList;
}


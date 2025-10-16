// src/math_foundations/matrix.rs
/*!
    A **row-major 2D matrix** abstraction on top of generic tensor backends.
        - Exposed **primary API**: `Matrix<T>` — a type-erased enum over Dense/Sparse backends.
        - Internal zero-cost wrapper: `MatrixGeneral<T, B>` for backend-specific utilities.
        - Backends: `crate::math::tensor::dense::Tensor` and `crate::math::tensor::sparse::Tensor`.

    This single file provides:
        - `Matrix<T>` (enum) with **elementwise arithmetic** (+, -, *, /) for
        Matrix ⊕ Matrix and Matrix ⊕ scalar (and the corresponding `op=` assigns).
        - Builders: `build_matrix`, `build_dense_matrix`, `build_sparse_matrix[_with]`.
        - Row/column helper views on `MatrixGeneral`.
        - A `prelude` module to import everything you need in external crates.
        - A `matrix_dispatch_same_backend!` macro to reduce boilerplate when
        matching on enum backends.
*/

use core::marker::PhantomData;
use core::ops::{
    Deref, DerefMut, Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign
};

use crate::math::scalar::Scalar;
use crate::math::tensor::tensor_trait::TensorTrait;
use crate::math::tensor::dense::Tensor as TensorDense;
use crate::math::tensor::sparse::Tensor as TensorSparse;

// ============================================================================
// ------------------------------- Core Structs --------------------------------
// ============================================================================

/// Zero-cost, rank-2 wrapper over a generic tensor backend.
/// Keep this for backend-specific helpers and internal composition.
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct MatrixGeneral<T: Scalar, B: TensorTrait<T>> {
    pub tensor: B,          // backend tensor (dense or sparse), shape = [rows, cols]
    _pd: PhantomData<T>,    // bind T; keep private
}

/// Type-erased matrix over either dense or sparse backend. **Primary exposed API.**
#[derive(Debug, Clone)]
pub enum Matrix<T: Scalar> {
    Dense(MatrixDense<T>),
    Sparse(MatrixSparse<T>),
}

/// Handy aliases (requested names).
pub type MatrixDense<T>  = MatrixGeneral<T, TensorDense<T>>;
pub type MatrixSparse<T> = MatrixGeneral<T, TensorSparse<T>>;

// ============================================================================
// ------------------------------- Views API ----------------------------------
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct RowRef<'a, T: Scalar, B: TensorTrait<T>> {
    mat: &'a MatrixGeneral<T, B>,
    row: isize,
}

#[derive(Clone, Copy, Debug)]
pub struct ColRef<'a, T: Scalar, B: TensorTrait<T>> {
    mat: &'a MatrixGeneral<T, B>,
    col: isize,
}

/// A mutable view of a **row** in a `MatrixGeneral`.
pub struct RowRefMut<'a, T: Scalar, B: TensorTrait<T>> {
    mat: &'a mut MatrixGeneral<T, B>,
    row: isize,
}

/// A mutable view of a **column** in a `MatrixGeneral`.
pub struct ColRefMut<'a, T: Scalar, B: TensorTrait<T>> {
    mat: &'a mut MatrixGeneral<T, B>,
    col: isize,
}

// ============================================================================
// ---------------------------- BackendKind & Builders ------------------------
// ============================================================================

/// Select which backend to build at runtime.
#[derive(Clone, Copy, Debug)]
pub enum BackendKind {
    Dense,
    Sparse,
}

/// Build a **dense** matrix from a row-major slice.
///
/// # Panics
/// Panics if `data.len() != rows * cols`.
#[inline]
pub fn build_dense_matrix<T>(rows: usize, cols: usize, data: &[T]) -> MatrixDense<T>
where
    T: Scalar + Scalar,
{
    assert!(
        data.len() == rows * cols,
        "build_dense_matrix: data length {} != rows*cols {}",
        data.len(),
        rows * cols
    );

    let mut t = TensorDense::<T>::empty(&[rows, cols]);
    for (idx, &v) in data.iter().enumerate() {
        let i = idx / cols;
        let j = idx % cols;
        t.set(&[i as isize, j as isize], v);
    }
    MatrixGeneral { tensor: t, _pd: PhantomData }
}

/// Build a **sparse** matrix from a row-major slice, treating exact zeros as implicit.
/// For floats, consider `build_sparse_matrix_with(..., |x: &f64| x.abs() < 1e-12)`.
#[inline]
pub fn build_sparse_matrix<T>(rows: usize, cols: usize, data: &[T]) -> MatrixSparse<T>
where
    T: Scalar + Scalar + PartialEq,
{
    build_sparse_matrix_with(rows, cols, data, |x| *x == T::zero())
}

/// Build a **sparse** matrix with a custom zero predicate.
///
/// # Panics
/// Panics if `data.len() != rows * cols`.
#[inline]
pub fn build_sparse_matrix_with<T, F>(
    rows: usize,
    cols: usize,
    data: &[T],
    is_zero: F,
) -> MatrixSparse<T>
where
    T: Scalar + Scalar,
    F: Fn(&T) -> bool,
{
    assert!(
        data.len() == rows * cols,
        "build_sparse_matrix_with: data length {} != rows*cols {}",
        data.len(),
        rows * cols
    );

    let mut t = TensorSparse::<T>::empty(&[rows, cols]);
    for (idx, &v) in data.iter().enumerate() {
        if !is_zero(&v) {
            let i = idx / cols;
            let j = idx % cols;
            t.set(&[i as isize, j as isize], v);
        }
    }
    MatrixGeneral { tensor: t, _pd: PhantomData }
}

/// One-stop builder selecting backend at runtime. Returns the **type-erased** `Matrix<T>`.
#[inline]
pub fn build_matrix<T>(
    rows: usize,
    cols: usize,
    data: &[T],
    kind: BackendKind,
) -> Matrix<T>
where
    T: Scalar + Scalar + PartialEq,
{
    match kind {
        BackendKind::Dense  => Matrix::Dense(build_dense_matrix(rows, cols, data)),
        BackendKind::Sparse => Matrix::Sparse(build_sparse_matrix(rows, cols, data)),
    }
}

// ============================================================================
// ------------------------------ MatrixGeneral I/O ---------------------------
// ============================================================================

impl<T: Scalar, B: TensorTrait<T>> MatrixGeneral<T, B> {
    /// Create a new `MatrixGeneral` with the given shape, filled with zeros.
    #[inline]
    pub fn empty(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "MatrixGeneral::new: shape must be nonzero");
        let tensor = B::empty(&[rows, cols]);
        Self { tensor, _pd: PhantomData }
    }

    /// Wrap an existing 2D tensor backend as a `MatrixGeneral`.
    #[inline]
    pub fn from_tensor(storage: B) -> Self {
        let shape = storage.shape();
        assert!(shape.len() == 2, "MatrixGeneral::from_tensor: tensor must be rank-2");
        let (rows, cols) = (shape[0], shape[1]);
        assert!(rows > 0 && cols > 0, "MatrixGeneral::from_tensor: shape must be nonzero");
        Self { tensor: storage, _pd: PhantomData }
    }

    /// Borrow the underlying tensor.
    #[inline] pub fn as_tensor(&self) -> &B { &self.tensor }
    #[inline] pub fn as_tensor_mut(&mut self) -> &mut B { &mut self.tensor }

    /// Type casting (delegates to backend).
    #[inline]
    pub fn cast_to<U>(&self) -> MatrixGeneral<U, B::Repr<U>>
    where
        U: Scalar,
    {
        let storage_u: B::Repr<U> = self.tensor.cast_to::<U>();
        MatrixGeneral { tensor: storage_u, _pd: PhantomData }
    }

    #[inline]
    pub fn print(&self) { self.tensor.print(); }
}

// ============================================================================
// ------------------------------ MatrixGeneral Accessors ---------------------
// ============================================================================

impl<T: Scalar, B: TensorTrait<T>> MatrixGeneral<T, B> {
    /// ----------------- Shape helpers ---------------------
    #[inline]
    pub fn rows(&self) -> usize {
        debug_assert_eq!(self.tensor.shape().len(), 2, "Matrix must be rank-2");
        self.tensor.shape()[0]
    }

    #[inline]
    pub fn cols(&self) -> usize {
        debug_assert_eq!(self.tensor.shape().len(), 2, "Matrix must be rank-2");
        self.tensor.shape()[1]
    }

    #[inline]
    pub fn shape(&self) -> &[usize] { self.tensor.shape() }

    /// --------------- Low-level accessors -----------------
    /// Get by value at (i, j) via backend (wrapped indices).
    #[inline]
    pub fn get(&self, i: isize, j: isize) -> T where T: Scalar { self.tensor.get(&[i, j]) }

    /// Set at (i, j) via backend (wrapped indices).
    #[inline]
    pub fn set(&mut self, i: isize, j: isize, val: T) { self.tensor.set(&[i, j], val) }

    /// Mutable element access (backend returns `&mut T`).
    #[inline]
    pub fn get_mut(&mut self, i: isize, j: isize) -> &mut T { self.tensor.get_mut(&[i, j]) }

    /// ----------------- High-level accessors -----------------
    #[inline] pub fn row(&self, i: isize) -> RowRef<'_, T, B> { RowRef { mat: self, row: i } }
    #[inline] pub fn col(&self, j: isize) -> ColRef<'_, T, B> { ColRef { mat: self, col: j } }
    #[inline] pub fn row_mut(&mut self, i: isize) -> RowRefMut<'_, T, B> { RowRefMut { mat: self, row: i } }
    #[inline] pub fn col_mut(&mut self, j: isize) -> ColRefMut<'_, T, B> { ColRefMut { mat: self, col: j } }
}

// ============================================================================
// --------------------------- MatrixGeneral Bulk Ops -------------------------
// ============================================================================

impl<T: Scalar, B: TensorTrait<T>> MatrixGeneral<T, B> {
    /// Set entire row `i` from `vals` (length must equal `cols()`).
    #[inline]
    pub fn set_row_from_slice(&mut self, i: isize, vals: &[T])
    where
        T: Scalar,
    {
        assert!(vals.len() == self.cols(), "set_row_from_slice: len mismatch");
        for (j, &v) in vals.iter().enumerate() {
            self.set(i, j as isize, v);
        }
    }

    /// Set entire column `j` from `vals` (length must equal `rows()`).
    #[inline]
    pub fn set_col_from_slice(&mut self, j: isize, vals: &[T])
    where
        T: Scalar,
    {
        assert!(vals.len() == self.rows(), "set_col_from_slice: len mismatch");
        for (i, &v) in vals.iter().enumerate() {
            self.set(i as isize, j, v);
        }
    }

    /// Fill using backend semantics (dense: all, sparse: stored nonzeros).
    #[inline]
    pub fn fill(&mut self, val: T)
    where
        T: Scalar,
    {
        self.tensor.par_fill(val);
    }
}

// ============================================================================
// ------------------------------- Views for MatrixGeneral --------------------
// ============================================================================

/// Immutable views
impl<'a, T: Scalar, B: TensorTrait<T>> RowRef<'a, T, B> {
    #[inline] pub fn len(&self) -> usize { self.mat.cols() }

    /// Get value at column `j` in this row.
    #[inline] pub fn get(&self, j: isize) -> T where T: Scalar {
        self.mat.get(self.row, j)
    }

    /// Scalar the row into `dst` (len must be `cols()`).
    #[inline]
    pub fn copy_to_slice(&self, dst: &mut [T])
    where
        T: Scalar,
    {
        assert!(
            dst.len() == self.len(),
            "RowRef::copy_to_slice: len mismatch (got {}, expected {})",
            dst.len(), self.len());
        for j in 0..self.len() {
            dst[j] = self.get(j as isize);
        }
    }

    /// Collect the row into a new `Vec<T>`.
    #[inline] pub fn to_vec(&self) -> Vec<T> where T: Scalar {
        let mut v = vec![T::zero(); self.len()];
        self.copy_to_slice(&mut v);
        v
    }
}

impl<'a, T: Scalar, B: TensorTrait<T>> ColRef<'a, T, B> {
    #[inline] pub fn len(&self) -> usize { self.mat.rows() }

    /// Get value at row `i` in this column.
    #[inline] pub fn get(&self, i: isize) -> T where T: Scalar {
        self.mat.get(i, self.col)
    }

    /// Scalar the column into `dst` (len must be `rows()`).
    #[inline]
    pub fn copy_to_slice(&self, dst: &mut [T])
    where
        T: Scalar,
    {
        assert!(
            dst.len() == self.len(),
            "ColRef::copy_to_slice: len mismatch (got {}, expected {})",
            dst.len(), self.len());
        for i in 0..self.len() {
            dst[i] = self.get(i as isize);
        }
    }

    /// Collect the column into a new `Vec<T>`.
    #[inline] pub fn to_vec(&self) -> Vec<T> where T: Scalar {
        let mut v = vec![T::zero(); self.len()];
        self.copy_to_slice(&mut v);
        v
    }
}

/// Mutable views
impl<'a, T: Scalar, B: TensorTrait<T>> RowRefMut<'a, T, B> {
    #[inline] pub fn len(&self) -> usize { self.mat.cols() }

    /// Set entry at column `j` in this row.
    #[inline] pub fn set(&mut self, j: isize, val: T) { self.mat.set(self.row, j, val); }

    /// Get a mutable ref at column `j` (useful for in-place ops).
    #[inline] pub fn get_mut(&mut self, j: isize) -> &mut T { self.mat.get_mut(self.row, j) }

    /// Fill the entire row with `val`.
    #[inline]
    pub fn fill(&mut self, val: T)
    where
        T: Scalar,
    {
        for j in 0..self.len() {
            self.set(j as isize, val);
        }
    }

    /// **Bulk set** the entire row from `vals` (length must equal `cols()`).
    #[inline]
    pub fn set_from_slice(&mut self, vals: &[T])
    where
        T: Scalar,
    {
        assert!(
            vals.len() == self.len(),
            "RowRefMut::set_from_slice: len mismatch (got {}, expected {})",
            vals.len(), self.len()
        );

        for (j, &v) in vals.iter().enumerate() {
            self.set(j as isize, v);
        }
    }
}

// Keep convenience deref to reach matrix methods while holding the row view.
impl<'a, T: Scalar, B: TensorTrait<T>> Deref for RowRefMut<'a, T, B> {
    type Target = MatrixGeneral<T, B>;
    #[inline] fn deref(&self) -> &Self::Target { self.mat }
}
impl<'a, T: Scalar, B: TensorTrait<T>> DerefMut for RowRefMut<'a, T, B> {
    #[inline] fn deref_mut(&mut self) -> &mut Self::Target { self.mat }
}

impl<'a, T: Scalar, B: TensorTrait<T>> ColRefMut<'a, T, B> {
    #[inline] pub fn len(&self) -> usize { self.mat.rows() }

    /// Set entry at row `i` in this column.
    #[inline] pub fn set(&mut self, i: isize, val: T) { self.mat.set(i, self.col, val); }

    /// Get a mutable ref at row `i`.
    #[inline] pub fn get_mut(&mut self, i: isize) -> &mut T { self.mat.get_mut(i, self.col) }

    /// Fill the entire column with `val`.
    #[inline]
    pub fn fill(&mut self, val: T)
    where
        T: Scalar,
    {
        for i in 0..self.len() {
            self.set(i as isize, val);
        }
    }

    /// **Bulk set** the entire column from `vals` (length must equal `rows()`).
    #[inline]
    pub fn set_from_slice(&mut self, vals: &[T])
    where
        T: Scalar,
    {
        assert!(
            vals.len() == self.len(),
            "ColRefMut::set_from_slice: len mismatch (got {}, expected {})",
            vals.len(), self.len()
        );
        for (i, &v) in vals.iter().enumerate() {
            self.set(i as isize, v);
        }
    }
}

impl<'a, T: Scalar, B: TensorTrait<T>> Deref for ColRefMut<'a, T, B> {
    type Target = MatrixGeneral<T, B>;
    #[inline] fn deref(&self) -> &Self::Target { self.mat }
}
impl<'a, T: Scalar, B: TensorTrait<T>> DerefMut for ColRefMut<'a, T, B> {
    #[inline] fn deref_mut(&mut self) -> &mut Self::Target { self.mat }
}

// ============================================================================
// ---------------------------- Enum Matrix (type-erased) ---------------------
// ============================================================================

impl<T: Scalar> Matrix<T> {
    // --------- shape / meta ----------
    #[inline] pub fn rows(&self) -> usize {
        match self {
            Matrix::Dense(m) => m.rows(),
            Matrix::Sparse(m) => m.rows(),
        }
    }
    #[inline] pub fn cols(&self) -> usize {
        match self {
            Matrix::Dense(m) => m.cols(),
            Matrix::Sparse(m) => m.cols(),
        }
    }
    #[inline] pub fn shape(&self) -> [usize; 2] { [self.rows(), self.cols()] }

    // --------- cloning / printing / casting ----------
    #[inline] pub fn print(&self) {
        match self {
            Matrix::Dense(m) => m.print(),
            Matrix::Sparse(m) => m.print(),
        }
    }

    /// Cast element type but **preserve backend** (dense→dense, sparse→sparse).
    #[inline]
    pub fn cast_to<U: Scalar>(&self) -> Matrix<U> {
        match self {
            Matrix::Dense(m) => Matrix::Dense(m.cast_to::<U>()),
            Matrix::Sparse(m) => Matrix::Sparse(m.cast_to::<U>()),
        }
    }

    // --------- element access ----------
    #[inline] pub fn get(&self, i: isize, j: isize) -> T where T: Scalar {
        match self {
            Matrix::Dense(m) => m.get(i, j),
            Matrix::Sparse(m) => m.get(i, j),
        }
    }
    #[inline] pub fn set(&mut self, i: isize, j: isize, v: T) {
        match self {
            Matrix::Dense(m) => m.set(i, j, v),
            Matrix::Sparse(m) => m.set(i, j, v),
        }
    }

    // --------- bulk helpers ----------
    #[inline] pub fn fill(&mut self, val: T) where T: Scalar {
        match self {
            Matrix::Dense(m) => m.fill(val),
            Matrix::Sparse(m) => m.fill(val),
        }
    }
    /// Scalar out a row as a fresh Vec (works for both backends).
    #[inline] pub fn row_to_vec(&self, i: isize) -> Vec<T> where T: Scalar {
        match self {
            Matrix::Dense(m) => m.row(i).to_vec(),
            Matrix::Sparse(m) => m.row(i).to_vec(),
        }
    }
    /// Scalar out a column as a fresh Vec (works for both backends).
    #[inline] pub fn col_to_vec(&self, j: isize) -> Vec<T> where T: Scalar {
        match self {
            Matrix::Dense(m) => m.col(j).to_vec(),
            Matrix::Sparse(m) => m.col(j).to_vec(),
        }
    }
    /// Set entire row from slice.
    #[inline] pub fn set_row_from_slice(&mut self, i: isize, vals: &[T]) where T: Scalar {
        match self {
            Matrix::Dense(m) => m.set_row_from_slice(i, vals),
            Matrix::Sparse(m) => m.set_row_from_slice(i, vals),
        }
    }
    /// Set entire column from slice.
    #[inline] pub fn set_col_from_slice(&mut self, j: isize, vals: &[T]) where T: Scalar {
        match self {
            Matrix::Dense(m) => m.set_col_from_slice(j, vals),
            Matrix::Sparse(m) => m.set_col_from_slice(j, vals),
        }
    }
}

// expose internal data as tensor
impl<T: Scalar> Matrix<T> {
    // -------- Borrowed accessors --------
    #[inline]
    pub fn as_dense_tensor(&self) -> Option<&TensorDense<T>> {
        match self {
            Matrix::Dense(m) => Some(&m.tensor),
            _ => None,
        }
    }

    #[inline]
    pub fn as_sparse_tensor(&self) -> Option<&TensorSparse<T>> {
        match self {
            Matrix::Sparse(m) => Some(&m.tensor),
            _ => None,
        }
    }

    // -------- Mutable borrowed accessors --------
    #[inline]
    pub fn as_dense_tensor_mut(&mut self) -> Option<&mut TensorDense<T>> {
        match self {
            Matrix::Dense(m) => Some(&mut m.tensor),
            _ => None,
        }
    }

    #[inline]
    pub fn as_sparse_tensor_mut(&mut self) -> Option<&mut TensorSparse<T>> {
        match self {
            Matrix::Sparse(m) => Some(&mut m.tensor),
            _ => None,
        }
    }

    // -------- Consuming accessors --------
    #[inline]
    pub fn into_dense_tensor(self) -> Option<TensorDense<T>> {
        match self {
            Matrix::Dense(m) => Some(m.tensor),
            _ => None,
        }
    }

    #[inline]
    pub fn into_sparse_tensor(self) -> Option<TensorSparse<T>> {
        match self {
            Matrix::Sparse(m) => Some(m.tensor),
            _ => None,
        }
    }
}


// ============================================================================
// -------------------- Macro: dispatch to same-backend variants --------------
// ============================================================================

/// Dispatch helper: run backend-specific blocks for **matching** backends.
/// Panics if the two matrices have different backends.
///
/// Usage:
/// ```ignore
/// let out = matrix_dispatch_same_backend!(
///     &a, &b,
///     |ad, bd| { /* dense body using MatrixDense<T> */ },
///     |as_, bs_| { /* sparse body using MatrixSparse<T> */ }
/// );
/// ```
macro_rules! matrix_dispatch_same_backend {
    ($lhs:expr, $rhs:expr,
     |$ad:ident, $bd:ident| $dense:block,
     |$as_:ident, $bs_:ident| $sparse:block
    ) => {{
        match ($lhs, $rhs) {
            (Matrix::Dense($ad),  Matrix::Dense($bd))  => $dense,
            (Matrix::Sparse($as_), Matrix::Sparse($bs_)) => $sparse,
            _ => panic!("Matrix: backend mismatch for binary operation"),
        }
    }};
}

// ============================================================================
// ----------------------------- Arithmetic Ops -------------------------------
// ============================================================================

// ---- Matrix ⊕ Matrix (by cloning LHS and zipping in-place) -----------------

macro_rules! impl_matrix_binop {
    ($trait:ident, $method:ident, $closure:expr, $bound:ident) => {
        impl<'a, T> $trait<&'a Matrix<T>> for &'a Matrix<T>
        where
            T: Scalar + core::ops::$bound<Output = T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: &'a Matrix<T>) -> Self::Output {
                matrix_dispatch_same_backend!(
                    self, rhs,
                    |ad, bd| {
                        let mut out = ad.clone();
                        out.tensor.par_zip_with_inplace(&bd.tensor, $closure);
                        Matrix::Dense(out)
                    },
                    |as_, bs_| {
                        let mut out = as_.clone();
                        out.tensor.par_zip_with_inplace(&bs_.tensor, $closure);
                        Matrix::Sparse(out)
                    }
                )
            }
        }
    };
}

macro_rules! impl_matrix_binop_assign {
    ($trait:ident, $method:ident, $closure:expr, $bound:ident) => {
        impl<T> $trait<&Matrix<T>> for Matrix<T>
        where
            T: Scalar + core::ops::$bound<Output = T>,
        {
            #[inline]
            fn $method(&mut self, rhs: &Matrix<T>) {
                match (self, rhs) {
                    (Matrix::Dense(a),  Matrix::Dense(b))  => {
                        a.tensor.par_zip_with_inplace(&b.tensor, $closure);
                    }
                    (Matrix::Sparse(a), Matrix::Sparse(b)) => {
                        a.tensor.par_zip_with_inplace(&b.tensor, $closure);
                    }
                    _ => panic!("Matrix: backend mismatch for assign operation"),
                }
            }
        }
    };
}

// +, -, *, /
impl_matrix_binop!(Add, add, |x, y| x + y, Add);
impl_matrix_binop!(Sub, sub, |x, y| x - y, Sub);
impl_matrix_binop!(Mul, mul, |x, y| x * y, Mul);
impl_matrix_binop!(Div, div, |x, y| x / y, Div);

impl_matrix_binop_assign!(AddAssign, add_assign, |x, y| x + y, Add);
impl_matrix_binop_assign!(SubAssign, sub_assign, |x, y| x - y, Sub);
impl_matrix_binop_assign!(MulAssign, mul_assign, |x, y| x * y, Mul);
impl_matrix_binop_assign!(DivAssign, div_assign, |x, y| x / y, Div);

// ---- &Matrix ⊕ scalar  and  Matrix op= scalar (map in place) ---------------

macro_rules! impl_matrix_scalar_binop {
    ($trait:ident, $method:ident, $op:tt, $bound:ident) => {
        impl<'a, T> $trait<T> for &'a Matrix<T>
        where
            T: Scalar + core::ops::$bound<Output = T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                match self {
                    Matrix::Dense(a) => {
                        let mut out = a.clone();
                        out.tensor.par_map_in_place(|x| x $op rhs);
                        Matrix::Dense(out)
                    }
                    Matrix::Sparse(a) => {
                        let mut out = a.clone();
                        out.tensor.par_map_in_place(|x| x $op rhs);
                        Matrix::Sparse(out)
                    }
                }
            }
        }
    };
}

macro_rules! impl_matrix_scalar_assign {
    ($trait:ident, $method:ident, $op:tt, $bound:ident) => {
        impl<T> $trait<T> for Matrix<T>
        where
            T: Scalar + core::ops::$bound<Output = T>,
        {
            #[inline]
            fn $method(&mut self, rhs: T) {
                match self {
                    Matrix::Dense(a) => a.tensor.par_map_in_place(|x| x $op rhs),
                    Matrix::Sparse(a) => a.tensor.par_map_in_place(|x| x $op rhs),
                }
            }
        }
    };
}

impl_matrix_scalar_binop!(Add, add, +, Add);
impl_matrix_scalar_binop!(Sub, sub, -, Sub);
impl_matrix_scalar_binop!(Mul, mul, *, Mul);
impl_matrix_scalar_binop!(Div, div, /, Div);

impl_matrix_scalar_assign!(AddAssign, add_assign, +, Add);
impl_matrix_scalar_assign!(SubAssign, sub_assign, -, Sub);
impl_matrix_scalar_assign!(MulAssign, mul_assign, *, Mul);
impl_matrix_scalar_assign!(DivAssign, div_assign, /, Div);



// ============================================================================
// --------------------------------- Prelude ----------------------------------
// ============================================================================

/// Bring this into scope with:
pub mod prelude {
    pub use super::{
        BackendKind,
        Matrix, MatrixDense, MatrixSparse, MatrixGeneral,
        RowRef, RowRefMut, ColRef, ColRefMut,
        build_matrix,
    };
}

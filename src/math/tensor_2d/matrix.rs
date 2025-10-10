// src/math_foundations/matrix.rs
/*!
A simple, contiguous **row-major** 2D tensor 
that other SoA or AoS types can build upon.

- Shape convention: `[rows, cols]`
- Storage: `Tensor<T>`
- Goals: minimal, safe, cache-friendly; clean row/col views

This file defines the core structs:
    - `Matrix`
    - `RowRef`, `RowRefMut`, 
    - `ColRef`, `ColRefMut` 
*/

use core::marker::PhantomData;
use core::ops::{
    Deref, DerefMut,
    Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign
};

use crate::math::scalar::Scalar;
use crate::math::tensor::tensor_trait::TensorTrait;

// ============================================================================
// ------------------------------- Core Structs -------------------------------
// ============================================================================

/// A 2D matrix wrapper over a generic tensor backend.
#[derive(Debug, Clone)]
pub struct Matrix<T: Scalar, B: TensorTrait<T>> {
    pub tensor: B, // backend tensor (dense or sparse), shape = [rows, cols]
    pub _pd: PhantomData<T>, // to bind T
}

/// Immutable views (many can coexist)
#[derive(Clone, Copy, Debug)]
pub struct RowRef<'a, T: Scalar, B: TensorTrait<T>> {
    mat: &'a Matrix<T, B>,
    row: isize,
}

#[derive(Clone, Copy, Debug)]
pub struct ColRef<'a, T: Scalar, B: TensorTrait<T>> {
    mat: &'a Matrix<T, B>,
    col: isize,
}

/// A mutable views.

pub struct RowRefMut<'a, T: Scalar, B: TensorTrait<T>> {
    mat: &'a mut Matrix<T, B>,
    row: isize,
}

/// A mutable view of a **column** in a `Matrix`.
pub struct ColRefMut<'a, T: Scalar, B: TensorTrait<T>> {
    mat: &'a mut Matrix<T, B>,
    col: isize,
}

// ============================================================================
// --------------------------------- Basic I/O --------------------------------
// ============================================================================

impl<T: Scalar, B: TensorTrait<T>> Matrix<T, B> {
    /// Create a new `Matrix` with the given shape, filled with zeros.
    #[inline]
    pub fn empty(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "Matrix::new: shape must be nonzero");
        let tensor = B::empty(&[rows, cols]);
        Self { tensor, _pd: PhantomData }
    }

    /// Wrap an existing 2D tensor backend as a `Matrix`.
    #[inline]
    pub fn from_tensor(storage: B) -> Self {
        let shape = storage.shape();
        assert!(shape.len() == 2, "Matrix::from_tensor: tensor must be rank-2");
        let (rows, cols) = (shape[0], shape[1]);
        assert!(rows > 0 && cols > 0, "Matrix::from_tensor: shape must be nonzero");
        Self { tensor: storage, _pd: PhantomData }
    }

    /// Borrow the underlying tensor.
    #[inline] pub fn as_tensor(&self) -> &B { &self.tensor }
    #[inline] pub fn as_tensor_mut(&mut self) -> &mut B { &mut self.tensor }

    // type casting
    pub fn cast_to<U>(&self) -> Matrix<U, B::Repr<U>>
    where
        T: Copy + Send + Sync,
        U: Scalar + Send + Sync,
    {
        let storage_u: B::Repr<U> = self.tensor.cast_to::<U>();
        Matrix { tensor: storage_u, _pd: PhantomData }
    }
}





// ============================================================================
// --------------------------------- Accessors --------------------------------
// ============================================================================

impl<T: Scalar, B: TensorTrait<T>> Matrix<T, B> {
    /// ----------------- Shape helpers ---------------------
    #[inline] pub fn rows(&self) -> usize { self.tensor.shape()[0] }
    #[inline] pub fn cols(&self) -> usize { self.tensor.shape()[0] }
    #[inline] pub fn shape(&self) -> &[usize] { self.tensor.shape() }

    /// --------------- Low-level accessors -----------------
    /// Get by value at (i, j) via backend.
    #[inline] pub fn get(&self, i: isize, j: isize) -> T { self.tensor.get(&[i, j])}

    /// Set at (i, j) via backend.
    #[inline] pub fn set(&mut self, i: isize, j: isize, val: T) { self.tensor.set(&[i, j], val) }

    /// Mutable element access (backend returns `&mut T`).
    #[inline] pub fn get_mut(&mut self, i: isize, j: isize) -> &mut T { self.tensor.get_mut(&[i, j])}

    /// ----------------- High-level accessors -----------------
    #[inline] pub fn row(&self, i: isize) -> RowRef<'_, T, B> { RowRef { mat: self, row: i } }
    #[inline] pub fn col(&self, j: isize) -> ColRef<'_, T, B> { ColRef { mat: self, col: j } }
    #[inline] pub fn row_mut(&mut self, i: isize) -> RowRefMut<'_, T, B> { RowRefMut { mat: self, row: i } }
    #[inline] pub fn col_mut(&mut self, j: isize) -> ColRefMut<'_, T, B> { ColRefMut { mat: self, col: j } }

}



// ============================================================================
// --------------------------- Bulk Operations --------------------------------
// ============================================================================
impl<T: Scalar, B: TensorTrait<T>> Matrix<T, B> {
    /// Set entire row `i` from `vals` (length must equal `cols()`).
    #[inline]
    pub fn set_row_from_slice(&mut self, i: isize, vals: &[T])
    where
        T: Copy,
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
        T: Copy,
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
        T: Copy + Send + Sync,
    {
        self.tensor.par_fill(val);
    }

}



// ============================================================================
// --------------------------------- Views ------------------------------------
// ============================================================================

/// Immutable views
impl<'a, T: Scalar, B: TensorTrait<T>> RowRef<'a, T, B> {
    #[inline] pub fn len(&self) -> usize { self.mat.cols() }

    /// Get value at column `j` in this row.
    #[inline] pub fn get(&self, j: isize) -> T where T: Copy {
        self.mat.get(self.row, j)
    }

    /// Copy the row into `dst` (len must be `cols()`).
    #[inline]
    pub fn copy_to_slice(&self, dst: &mut [T])
    where
        T: Copy,
    {
        assert!(dst.len() == self.len(), "RowRef::copy_to_slice: len mismatch");
        for j in 0..self.len() {
            dst[j] = self.get(j as isize);
        }
    }

    /// Collect the row into a new `Vec<T>`.
    #[inline] pub fn to_vec(&self) -> Vec<T> where T: Copy {
        let mut v = vec![T::zero(); self.len()];
        self.copy_to_slice(&mut v);
        v
    }
}

impl<'a, T: Scalar, B: TensorTrait<T>> ColRef<'a, T, B> {
    #[inline] pub fn len(&self) -> usize { self.mat.rows() }

    /// Get value at row `i` in this column.
    #[inline] pub fn get(&self, i: isize) -> T where T: Copy {
        self.mat.get(i, self.col)
    }

    /// Copy the column into `dst` (len must be `rows()`).
    #[inline]
    pub fn copy_to_slice(&self, dst: &mut [T])
    where
        T: Copy,
    {
        assert!(dst.len() == self.len(), "ColRef::copy_to_slice: len mismatch");
        for i in 0..self.len() {
            dst[i] = self.get(i as isize);
        }
    }

    /// Collect the column into a new `Vec<T>`.
    #[inline] pub fn to_vec(&self) -> Vec<T> where T: Copy {
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
        T: Copy,
    {
        for j in 0..self.len() {
            self.set(j as isize, val);
        }
    }

    /// **Bulk set** the entire row from `vals` (length must equal `cols()`).
    #[inline]
    pub fn set_from_slice(&mut self, vals: &[T])
    where
        T: Copy,
    {
        assert!(vals.len() == self.len(), "RowRefMut::set_from_slice: len mismatch");
        for (j, &v) in vals.iter().enumerate() {
            self.set(j as isize, v);
        }
    }
}

// Keep convenience deref to reach matrix methods while holding the row view.
impl<'a, T: Scalar, B: TensorTrait<T>> Deref for RowRefMut<'a, T, B> {
    type Target = Matrix<T, B>;
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
        T: Copy,
    {
        for i in 0..self.len() {
            self.set(i as isize, val);
        }
    }

    /// **Bulk set** the entire column from `vals` (length must equal `rows()`).
    #[inline]
    pub fn set_from_slice(&mut self, vals: &[T])
    where
        T: Copy,
    {
        assert!(vals.len() == self.len(), "ColRefMut::set_from_slice: len mismatch");
        for (i, &v) in vals.iter().enumerate() {
            self.set(i as isize, v);
        }
    }
}

impl<'a, T: Scalar, B: TensorTrait<T>> Deref for ColRefMut<'a, T, B> {
    type Target = Matrix<T, B>;
    #[inline] fn deref(&self) -> &Self::Target { self.mat }
}
impl<'a, T: Scalar, B: TensorTrait<T>> DerefMut for ColRefMut<'a, T, B> {
    #[inline] fn deref_mut(&mut self) -> &mut Self::Target { self.mat }
}



// ============================================================================
// --------------------------- Arithmetic Ops ---------------------------------
// ============================================================================

// ------------------- &Matrix ⊕ &Matrix -> Matrix -------------------

impl<'a, T, B> Add<&'a Matrix<T, B>> for &'a Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    // backend provides: &B + &B -> B
    for<'b> &'b B: Add<&'b B, Output = B>,
{
    type Output = Matrix<T, B>;
    #[inline]
    fn add(self, rhs: &'a Matrix<T, B>) -> Self::Output {
        let tensor = self.as_tensor() + rhs.as_tensor();
        Matrix { tensor, _pd: PhantomData }
    }
}

impl<'a, T, B> Sub<&'a Matrix<T, B>> for &'a Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Sub<&'b B, Output = B>,
{
    type Output = Matrix<T, B>;
    #[inline]
    fn sub(self, rhs: &'a Matrix<T, B>) -> Self::Output {
        let tensor = self.as_tensor() - rhs.as_tensor();
        Matrix { tensor, _pd: PhantomData }
    }
}

impl<'a, T, B> Mul<&'a Matrix<T, B>> for &'a Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Mul<&'b B, Output = B>,
{
    type Output = Matrix<T, B>;
    #[inline]
    fn mul(self, rhs: &'a Matrix<T, B>) -> Self::Output {
        let tensor = self.as_tensor() * rhs.as_tensor();
        Matrix { tensor, _pd: PhantomData }
    }
}

impl<'a, T, B> Div<&'a Matrix<T, B>> for &'a Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Div<&'b B, Output = B>,
{
    type Output = Matrix<T, B>;
    #[inline]
    fn div(self, rhs: &'a Matrix<T, B>) -> Self::Output {
        let tensor = self.as_tensor() / rhs.as_tensor();
        Matrix { tensor, _pd: PhantomData }
    }
}

// ------------------- Matrix op= Matrix -------------------

impl<T, B> AddAssign<&Matrix<T, B>> for Matrix<T, B>
where
    T: Scalar + Copy + Send + Sync + core::ops::Add<Output = T>,
    B: TensorTrait<T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: &Matrix<T, B>) {
        self.as_tensor_mut()
            .par_zip_with_inplace(rhs.as_tensor(), |x, y| x + y);
    }
}

impl<T, B> SubAssign<&Matrix<T, B>> for Matrix<T, B>
where
    T: Scalar + Copy + Send + Sync + core::ops::Sub<Output = T>,
    B: TensorTrait<T>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &Matrix<T, B>) {
        self.as_tensor_mut()
            .par_zip_with_inplace(rhs.as_tensor(), |x, y| x - y);
    }
}

impl<T, B> MulAssign<&Matrix<T, B>> for Matrix<T, B>
where
    T: Scalar + Copy + Send + Sync + core::ops::Mul<Output = T>,
    B: TensorTrait<T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &Matrix<T, B>) {
        self.as_tensor_mut()
            .par_zip_with_inplace(rhs.as_tensor(), |x, y| x * y);
    }
}

impl<T, B> DivAssign<&Matrix<T, B>> for Matrix<T, B>
where
    T: Scalar + Copy + Send + Sync + core::ops::Div<Output = T>,
    B: TensorTrait<T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: &Matrix<T, B>) {
        self.as_tensor_mut()
            .par_zip_with_inplace(rhs.as_tensor(), |x, y| x / y);
    }
}

// ------------------- &Matrix ⊕ scalar -> Matrix -------------------

impl<'a, T, B> Add<T> for &'a Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Add<T, Output = B>,
{
    type Output = Matrix<T, B>;
    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        let tensor = self.as_tensor() + rhs;
        Matrix { tensor, _pd: PhantomData }
    }
}

impl<'a, T, B> Sub<T> for &'a Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Sub<T, Output = B>,
{
    type Output = Matrix<T, B>;
    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        let tensor = self.as_tensor() - rhs;
        Matrix { tensor, _pd: PhantomData }
    }
}

impl<'a, T, B> Mul<T> for &'a Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Mul<T, Output = B>,
{
    type Output = Matrix<T, B>;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        let tensor = self.as_tensor() * rhs;
        Matrix { tensor, _pd: PhantomData }
    }
}

impl<'a, T, B> Div<T> for &'a Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Div<T, Output = B>,
{
    type Output = Matrix<T, B>;
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        let tensor = self.as_tensor() / rhs;
        Matrix { tensor, _pd: PhantomData }
    }
}

// ------------------- Matrix scalar-op -------------------

impl<T, B> AddAssign<T> for Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T> + AddAssign<T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.as_tensor_mut().add_assign(rhs);
    }
}

impl<T, B> SubAssign<T> for Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T> + SubAssign<T>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.as_tensor_mut().sub_assign(rhs);
    }
}

impl<T, B> MulAssign<T> for Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T> + MulAssign<T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.as_tensor_mut().mul_assign(rhs);
    }
}

impl<T, B> DivAssign<T> for Matrix<T, B>
where
    T: Scalar,
    B: TensorTrait<T> + DivAssign<T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.as_tensor_mut().div_assign(rhs);
    }
}

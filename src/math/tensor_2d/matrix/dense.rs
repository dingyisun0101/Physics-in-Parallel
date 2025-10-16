/*!
Unified **dense 2D matrix** over a row-major `dense::Tensor<T>` backend,
configurable as logically **Row-major** or **Col-major**.

- Major direction (Row or Col) returns zero-copy slices.
- Non-major direction exposes cached mutable guards elsewhere if needed; however,
  the **MatrixTrait** requires true zero-copy `&[T]`/`&mut [T]` views, so the
  non-major view methods here **panic** (by design) to enforce the contract.

This file provides:
- `Matrix<T>` with `Major::{Row, Col}`
- View helpers (`get_row_ref{,_mut}`, `get_col_ref{,_mut}`) that can return cached
  guards for the non-major direction (not part of the trait).
- A clean `impl MatrixTrait<T> for Matrix<T>` implementing the required API.
*/

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use crate::math::{
    scalar::Scalar,
    tensor::{
        tensor_trait::TensorTrait,
        dense::Tensor,
    },
    tensor_2d::matrix::matrix_trait::MatrixTrait,
};

// =============================================================================
// --------------------------------- Views -------------------------------------
// =============================================================================

/// Zero-copy borrowed slice view (contiguous).
#[derive(Clone, Copy, Debug)]
pub struct SliceView<'a, T> {
    pub slice: &'a [T],
}
impl<'a, T> Deref for SliceView<'a, T> {
    type Target = [T];
    #[inline] fn deref(&self) -> &Self::Target { self.slice }
}

/// Cached view for non-contiguous direction (owns a buffer).
#[derive(Clone, Debug)]
pub struct SliceViewCached<T> {
    pub buf: Vec<T>,
}
impl<T> Deref for SliceViewCached<T> {
    type Target = [T];
    #[inline] fn deref(&self) -> &Self::Target { &self.buf }
}

/// Unified immutable row/column view. Always derefs to `&[T]`.
#[derive(Clone, Debug)]
pub enum MatrixSlice<'a, T> {
    Borrowed(SliceView<'a, T>),
    Cached(SliceViewCached<T>),
}
impl<'a, T> Deref for MatrixSlice<'a, T> {
    type Target = [T];
    #[inline] fn deref(&self) -> &Self::Target {
        match self {
            MatrixSlice::Borrowed(v) => v,
            MatrixSlice::Cached(v) => v,
        }
    }
}

/// Guard that materializes a non-major mutable view and **commits on drop**.
pub struct NonMajorMutGuard<'a, T: Scalar> {
    parent: *mut Matrix<T>,   // raw to satisfy Drop while holding &mut self
    idx: usize,               // row index if is_row, else column index
    is_row: bool,             // true => row, false => col
    buf: Vec<T>,              // working buffer exposed via DerefMut
    _pd: PhantomData<&'a mut Matrix<T>>,
}
impl<'a, T: Scalar> Deref for NonMajorMutGuard<'a, T> {
    type Target = [T];
    #[inline] fn deref(&self) -> &Self::Target { &self.buf }
}
impl<'a, T: Scalar> DerefMut for NonMajorMutGuard<'a, T> {
    #[inline] fn deref_mut(&mut self) -> &mut Self::Target { &mut self.buf }
}
impl<'a, T: Scalar> Drop for NonMajorMutGuard<'a, T> {
    fn drop(&mut self) {
        // Commit edits back to the parent matrix.
        // SAFETY: The guard is created from &mut Matrix<T> and lives as long as that borrow.
        let m = unsafe { &mut *self.parent };
        if self.is_row {
            debug_assert_eq!(self.buf.len(), m.cols);
            match m.major {
                Major::Row => unreachable!("NonMajorMutGuard(row) not used for Row-major"),
                Major::Col => {
                    let r = self.idx;
                    for (j, &v) in self.buf.iter().enumerate() {
                        // logical (r, j) -> physical (j, r)
                        m.tensor.set(&[j as isize, r as isize], v);
                    }
                }
            }
        } else {
            debug_assert_eq!(self.buf.len(), m.rows);
            match m.major {
                Major::Col => unreachable!("NonMajorMutGuard(col) not used for Col-major"),
                Major::Row => {
                    let c = self.idx;
                    for (i, &v) in self.buf.iter().enumerate() {
                        // logical (i, c) -> physical (i, c)
                        m.tensor.set(&[i as isize, c as isize], v);
                    }
                }
            }
        }
    }
}

/// Unified mutable view: borrowed &mut slice when contiguous; guard otherwise.
pub enum MatrixSliceMut<'a, T: Scalar> {
    Borrowed(&'a mut [T]),
    Guard(NonMajorMutGuard<'a, T>),
}
impl<'a, T: Scalar> Deref for MatrixSliceMut<'a, T> {
    type Target = [T];
    #[inline] fn deref(&self) -> &Self::Target {
        match self {
            MatrixSliceMut::Borrowed(s) => s,
            MatrixSliceMut::Guard(g) => g,
        }
    }
}
impl<'a, T: Scalar> DerefMut for MatrixSliceMut<'a, T> {
    #[inline] fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            MatrixSliceMut::Borrowed(s) => s,
            MatrixSliceMut::Guard(g) => g,
        }
    }
}

// =============================================================================
/*                                Matrix (dense)                              */
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Major { Row, Col }

#[derive(Debug, Clone)]
pub struct Matrix<T: Scalar> {
    tensor: Tensor<T>, // physical storage
    major: Major,      // logical major
    rows: usize,       // logical rows
    cols: usize,       // logical cols
}

#[inline(always)]
fn wrap_axis_index(idx: isize, dim: usize) -> usize {
    debug_assert!(dim > 0);
    let d = dim as isize;
    let mut m = idx % d;
    if m < 0 { m += d; }
    m as usize
}

impl<T: Scalar> Matrix<T> {
    // -------------------------------------------------------------------------
    // Constructors / backend accessors
    // -------------------------------------------------------------------------

    #[inline]
    pub fn empty(rows: usize, cols: usize, major: Major) -> Self {
        assert!(rows > 0 && cols > 0, "Matrix::empty: shape must be nonzero");
        let physical = match major { Major::Row => [rows, cols], Major::Col => [cols, rows] };
        let tensor = Tensor::<T>::empty(&physical);
        Self { tensor, major, rows, cols }
    }

    #[inline]
    pub fn from_tensor(src: impl TensorTrait<T>, major: Major) -> Self {
        let shape = src.shape();
        assert!(shape.len() == 2, "Matrix::from_tensor: source must be rank-2");
        let (rows, cols) = (shape[0], shape[1]);
        let mut m = Self::empty(rows, cols, major);
        for i in 0..rows {
            for j in 0..cols {
                let v = src.get(&[i as isize, j as isize]);
                m.set(i as isize, j as isize, v);
            }
        }
        m
    }

    #[inline] pub fn major(&self) -> Major { self.major }
    #[inline] pub fn rows(&self) -> usize { self.rows }
    #[inline] pub fn cols(&self) -> usize { self.cols }
    #[inline] pub fn shape(&self) -> [usize; 2] { [self.rows, self.cols] }
    #[inline] pub fn backend(&self) -> &Tensor<T> { &self.tensor }
    #[inline] pub fn backend_mut(&mut self) -> &mut Tensor<T> { &mut self.tensor }

    #[inline]
    fn assert_compat(&self, rhs: &Self) {
        assert_eq!(self.major, rhs.major, "Matrix layout mismatch: {:?} vs {:?}", self.major, rhs.major);
        assert_eq!(self.rows, rhs.rows, "Matrix row mismatch: {} vs {}", self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols, "Matrix col mismatch: {} vs {}", self.cols, rhs.cols);
    }

    /// Build directly from a physical dense tensor with explicit logical layout.
    #[inline]
    pub fn from_backend_with_layout(
        tensor: Tensor<T>,
        major: Major,
        rows: usize,
        cols: usize,
    ) -> Self {
        let shp = tensor.shape();
        match major {
            Major::Row => assert!(shp == [rows, cols], "backend shape {:?} != [{}, {}]", shp, rows, cols),
            Major::Col => assert!(shp == [cols, rows], "backend shape {:?} != [{}, {}] (col-major uses [cols, rows])", shp, cols, rows),
        }
        Self { tensor, major, rows, cols }
    }

    /// Cast element type while preserving **logical** shape and major layout.
    #[inline]
    pub fn cast_to<U: Scalar>(&self) -> Matrix<U> {
        let t_u: Tensor<U> = self.backend().cast_to::<U>();
        Matrix::<U>::from_backend_with_layout(t_u, self.major, self.rows, self.cols)
    }

    // -------------------------------------------------------------------------
    // View helpers (non-trait): immutable & mutable references / guards
    // -------------------------------------------------------------------------

    #[inline]
    pub fn get(&self, i: isize, j: isize) -> T where T: Copy {
        match self.major {
            Major::Row => self.tensor.get(&[i, j]),
            Major::Col => self.tensor.get(&[j, i]),
        }
    }

    #[inline]
    pub fn set(&mut self, i: isize, j: isize, val: T) {
        match self.major {
            Major::Row => self.tensor.set(&[i, j], val),
            Major::Col => self.tensor.set(&[j, i], val),
        }
    }

    #[inline]
    pub fn get_row_ref<'a>(&'a self, i: isize) -> MatrixSlice<'a, T>
    where
        T: Copy,
    {
        let r = wrap_axis_index(i, self.rows);
        match self.major {
            Major::Row => {
                let off = r * self.cols;
                MatrixSlice::Borrowed(SliceView { slice: &self.tensor.data[off .. off + self.cols] })
            }
            Major::Col => {
                let mut buf = Vec::with_capacity(self.cols);
                for j in 0..self.cols {
                    buf.push(self.tensor.get(&[j as isize, r as isize]));
                }
                MatrixSlice::Cached(SliceViewCached { buf })
            }
        }
    }

    #[inline]
    pub fn get_col_ref<'a>(&'a self, j: isize) -> MatrixSlice<'a, T>
    where
        T: Copy,
    {
        let c = wrap_axis_index(j, self.cols);
        match self.major {
            Major::Col => {
                let off = c * self.rows;
                MatrixSlice::Borrowed(SliceView { slice: &self.tensor.data[off .. off + self.rows] })
            }
            Major::Row => {
                let mut buf = Vec::with_capacity(self.rows);
                for i in 0..self.rows {
                    buf.push(self.tensor.get(&[i as isize, c as isize]));
                }
                MatrixSlice::Cached(SliceViewCached { buf })
            }
        }
    }

    #[inline]
    pub fn get_row_ref_mut<'a>(&'a mut self, i: isize) -> MatrixSliceMut<'a, T>
    where
        T: Copy,
    {
        let r = wrap_axis_index(i, self.rows);
        match self.major {
            Major::Row => {
                let off = r * self.cols;
                MatrixSliceMut::Borrowed(&mut self.tensor.data[off .. off + self.cols])
            }
            Major::Col => {
                let mut buf = Vec::with_capacity(self.cols);
                for j in 0..self.cols {
                    buf.push(self.tensor.get(&[j as isize, r as isize]));
                }
                MatrixSliceMut::Guard(NonMajorMutGuard {
                    parent: self as *mut _,
                    idx: r,
                    is_row: true,
                    buf,
                    _pd: PhantomData,
                })
            }
        }
    }

    #[inline]
    pub fn get_col_ref_mut<'a>(&'a mut self, j: isize) -> MatrixSliceMut<'a, T>
    where
        T: Copy,
    {
        let c = wrap_axis_index(j, self.cols);
        match self.major {
            Major::Col => {
                let off = c * self.rows;
                MatrixSliceMut::Borrowed(&mut self.tensor.data[off .. off + self.rows])
            }
            Major::Row => {
                let mut buf = Vec::with_capacity(self.rows);
                for i in 0..self.rows {
                    buf.push(self.tensor.get(&[i as isize, c as isize]));
                }
                MatrixSliceMut::Guard(NonMajorMutGuard {
                    parent: self as *mut _,
                    idx: c,
                    is_row: false,
                    buf,
                    _pd: PhantomData,
                })
            }
        }
    }
}

// =============================================================================
/*                           MatrixTrait implementation                        */
// =============================================================================

impl<T: Scalar> MatrixTrait<T> for Matrix<T> {
    // ----------------------------------- Shape -------------------------------
    #[inline]
    fn rows(&self) -> usize { self.rows }
    #[inline]
    fn cols(&self) -> usize { self.cols }

    // ----------------------------- Element access ----------------------------
    #[inline]
    fn get(&self, i: isize, j: isize) -> T {
        match self.major {
            Major::Row => self.tensor.get(&[i, j]),
            Major::Col => self.tensor.get(&[j, i]),
        }
    }

    #[inline]
    fn set(&mut self, i: isize, j: isize, val: T) {
        match self.major {
            Major::Row => self.tensor.set(&[i, j], val),
            Major::Col => self.tensor.set(&[j, i], val),
        }
    }

    // ------------------------------ Zero-copy views --------------------------
    #[inline]
    fn row_view<'a>(&'a self, i: isize) -> &'a [T] {
        let r = wrap_axis_index(i, self.rows);
        match self.major {
            Major::Row => {
                let off = r * self.cols;
                &self.tensor.data[off .. off + self.cols]
            }
            Major::Col => panic!("row_view: cannot return contiguous row slice in Col-major layout"),
        }
    }

    #[inline]
    fn row_view_mut<'a>(&'a mut self, i: isize) -> &'a mut [T] {
        let r = wrap_axis_index(i, self.rows);
        match self.major {
            Major::Row => {
                let off = r * self.cols;
                &mut self.tensor.data[off .. off + self.cols]
            }
            Major::Col => panic!("row_view_mut: cannot return contiguous row slice in Col-major layout"),
        }
    }

    #[inline]
    fn col_view<'a>(&'a self, j: isize) -> &'a [T] {
        let c = wrap_axis_index(j, self.cols);
        match self.major {
            Major::Col => {
                let off = c * self.rows;
                &self.tensor.data[off .. off + self.rows]
            }
            Major::Row => panic!("col_view: cannot return contiguous column slice in Row-major layout"),
        }
    }

    #[inline]
    fn col_view_mut<'a>(&'a mut self, j: isize) -> &'a mut [T] {
        let c = wrap_axis_index(j, self.cols);
        match self.major {
            Major::Col => {
                let off = c * self.rows;
                &mut self.tensor.data[off .. off + self.rows]
            }
            Major::Row => panic!("col_view_mut: cannot return contiguous column slice in Row-major layout"),
        }
    }

    // ------------------------------- Bulk helpers ---------------------------
    #[inline]
    fn set_row_from_slice(&mut self, i: isize, vals: &[T]) {
        assert_eq!(vals.len(), self.cols, "set_row_from_slice: len mismatch");
        let dst = self.row_view_mut(i);
        dst.copy_from_slice(vals);
    }

    #[inline]
    fn set_col_from_slice(&mut self, j: isize, vals: &[T]) {
        assert_eq!(vals.len(), self.rows, "set_col_from_slice: len mismatch");
        let dst = self.col_view_mut(j);
        dst.copy_from_slice(vals);
    }

    // ---------------------------------- I/O ---------------------------------
    #[inline]
    fn print(&self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let v = self.get(i as isize, j as isize);
                print!("{:<8} ", v);
            }
            println!();
        }
    }

    // ------------------------------- Basic Linalg ----------------------------
    /// Logical transpose: `self <- self^T`. Preserves current `major`.
    #[inline]
    fn transpose(&mut self)
    where
        T: Copy,
    {
        let (rows, cols) = (self.rows, self.cols);

        // Build a new backend with the proper **physical** shape for current layout.
        let mut new_tensor = match self.major {
            Major::Row => Tensor::<T>::empty(&[cols, rows]),
            Major::Col => Tensor::<T>::empty(&[rows, cols]),
        };

        // Copy using logical accessors.
        for i in 0..rows {
            for j in 0..cols {
                let v = self.get(i as isize, j as isize);
                match self.major {
                    Major::Row => new_tensor.set(&[j as isize, i as isize], v),
                    Major::Col => new_tensor.set(&[i as isize, j as isize], v),
                }
            }
        }

        // Install and swap logical dims (layout/contiguity invariants preserved).
        self.tensor = new_tensor;
        self.rows = cols;
        self.cols = rows;
    }

    /// Clip each element to `[min_val, max_val]` using an in-place parallel map.
    #[inline]
    fn clamp(&mut self, min_val: T, max_val: T)
    where
        T: Scalar + PartialOrd,
    {
        debug_assert!(min_val <= max_val, "clip: min_val must be <= max_val");
        self.tensor.par_map_in_place(|x| {
            let mut y = x;
            if y < min_val { y = min_val; }
            if y > max_val { y = max_val; }
            y
        });
    }

    /// Normalize the whole matrix to **unit L2 norm**:
    /// `self <- self / sqrt(sum(x_i^2))` (no-op if norm is zero).
    #[inline]
    fn normalize(&mut self)
    where
        T: Copy + Send + Sync + PartialEq
          + core::ops::Add<Output = T>
          + core::ops::Mul<Output = T>
          + core::ops::Div<Output = T>,
    {
        // Accumulate sum of squares on the current buffer.
        let mut sum = T::zero();
        for &x in self.tensor.data.iter() {
            sum = sum + x * x;
        }
        let n = <T>::sqrt(sum);
        if n == T::zero() { return; }
        self.tensor.par_map_in_place(|x| x / n);
    }

    /// Normalize by the **maximum magnitude** (computed as `sqrt(max(x^2))`):
    /// `self <- self / max_norm` (no-op if all zeros).
    #[inline]
    fn normalize_by_max(&mut self)
    where
        T: Scalar + PartialOrd
          + core::ops::Mul<Output = T>
          + core::ops::Div<Output = T>,
    {
        let mut max_sq = T::zero();
        for &x in self.tensor.data.iter() {
            let xsq = x * x;
            if xsq > max_sq { max_sq = xsq; }
        }
        let m = <T>::sqrt(max_sq);
        if m == T::zero() { return; }
        self.tensor.par_map_in_place(|x| x / m);
    }
}

// =============================================================================
// ------------------------------- Arithmetic Ops ------------------------------
// =============================================================================

// &Matrix ⊕ &Matrix -> Matrix  (layout + shape must match)
macro_rules! impl_matrix_ref_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> core::ops::$trait<&'a Matrix<T>> for &'a Matrix<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<Output = T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: &'a Matrix<T>) -> Self::Output {
                self.assert_compat(rhs);
                let out = &self.backend().clone() $op rhs.backend();
                Matrix::<T>::from_backend_with_layout(out, self.major, self.rows, self.cols)
            }
        }
    };
}
impl_matrix_ref_binop!(Add, add, +);
impl_matrix_ref_binop!(Sub, sub, -);
impl_matrix_ref_binop!(Mul, mul, *);
impl_matrix_ref_binop!(Div, div, /);

// Matrix ⊕= &Matrix (in-place)
macro_rules! impl_matrix_ref_assign {
    ($trait:ident, $method:ident) => {
        impl<'a, T> core::ops::$trait<&'a Matrix<T>> for Matrix<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<T>,
        {
            #[inline]
            fn $method(&mut self, rhs: &'a Matrix<T>) {
                self.assert_compat(rhs);
                self.backend_mut().$method(rhs.backend());
            }
        }
    };
}
impl_matrix_ref_assign!(AddAssign, add_assign);
impl_matrix_ref_assign!(SubAssign, sub_assign);
impl_matrix_ref_assign!(MulAssign, mul_assign);
impl_matrix_ref_assign!(DivAssign, div_assign);

// &Matrix ⊕ scalar -> Matrix
macro_rules! impl_matrix_ref_scalar_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> core::ops::$trait<T> for &'a Matrix<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<Output = T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                let out = &self.backend().clone() $op rhs;
                Matrix::<T>::from_backend_with_layout(out, self.major, self.rows, self.cols)
            }
        }
    };
}
impl_matrix_ref_scalar_binop!(Add, add, +);
impl_matrix_ref_scalar_binop!(Sub, sub, -);
impl_matrix_ref_scalar_binop!(Mul, mul, *);
impl_matrix_ref_scalar_binop!(Div, div, /);

// Matrix ⊕= scalar (in-place)
macro_rules! impl_matrix_scalar_assign {
    ($trait:ident, $method:ident) => {
        impl<T> core::ops::$trait<T> for Matrix<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<T>,
        {
            #[inline]
            fn $method(&mut self, rhs: T) {
                self.backend_mut().$method(rhs);
            }
        }
    };
}
impl_matrix_scalar_assign!(AddAssign, add_assign);
impl_matrix_scalar_assign!(SubAssign, sub_assign);
impl_matrix_scalar_assign!(MulAssign, mul_assign);
impl_matrix_scalar_assign!(DivAssign, div_assign);

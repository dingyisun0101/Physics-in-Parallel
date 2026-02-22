/*!
Dense 2D matrix built on the core `tensor::rank_2::dense::Tensor2D` backend.

Layout policy
-------------
- Fixed row-major physical storage.
- Rows are contiguous and zero-copy.
- Columns are strided and exposed via cached/guarded helper views.
*/

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use ndarray::Array2;

use crate::math::{
    ndarray_convert::NdarrayConvert,
    scalar::Scalar,
    tensor::{
        core::dense::Tensor as DenseTensor,
        core::tensor_trait::TensorTrait,
        rank_2::dense::Tensor2D,
    },
};
use super::matrix_trait::MatrixTrait;

// =============================================================================
// --------------------------------- Views -------------------------------------
// =============================================================================

#[derive(Clone, Copy, Debug)]
pub struct SliceView<'a, T> {
    pub slice: &'a [T],
}
impl<'a, T> Deref for SliceView<'a, T> {
    type Target = [T];
    #[inline] fn deref(&self) -> &Self::Target { self.slice }
}

#[derive(Clone, Debug)]
pub struct SliceViewCached<T> {
    pub buf: Vec<T>,
}
impl<T> Deref for SliceViewCached<T> {
    type Target = [T];
    #[inline] fn deref(&self) -> &Self::Target { &self.buf }
}

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

/// Guard that materializes a mutable column view and commits on drop.
pub struct ColMutGuard<'a, T: Scalar> {
    parent: *mut Matrix<T>,
    col: usize,
    buf: Vec<T>,
    _pd: PhantomData<&'a mut Matrix<T>>,
}
impl<'a, T: Scalar> Deref for ColMutGuard<'a, T> {
    type Target = [T];
    #[inline] fn deref(&self) -> &Self::Target { &self.buf }
}
impl<'a, T: Scalar> DerefMut for ColMutGuard<'a, T> {
    #[inline] fn deref_mut(&mut self) -> &mut Self::Target { &mut self.buf }
}
impl<'a, T: Scalar> Drop for ColMutGuard<'a, T> {
    fn drop(&mut self) {
        // SAFETY: guard is created from `&mut Matrix<T>` and cannot outlive it.
        let m = unsafe { &mut *self.parent };
        debug_assert_eq!(self.buf.len(), m.rows);
        for (i, &v) in self.buf.iter().enumerate() {
            m.tensor.set(i as isize, self.col as isize, v);
        }
    }
}

pub enum MatrixSliceMut<'a, T: Scalar> {
    Borrowed(&'a mut [T]),
    Guard(ColMutGuard<'a, T>),
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
// ------------------------------ Matrix (dense) -------------------------------
// =============================================================================

#[derive(Debug, Clone)]
pub struct Matrix<T: Scalar> {
    tensor: Tensor2D<T>,
    rows: usize,
    cols: usize,
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
    #[inline]
    pub fn empty(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "Matrix::empty: shape must be nonzero");
        let tensor = Tensor2D::<T>::empty(rows, cols);
        Self { tensor, rows, cols }
    }

    #[inline]
    pub fn from_tensor(src: impl TensorTrait<T>) -> Self {
        let shape = src.shape();
        assert!(shape.len() == 2, "Matrix::from_tensor: source must be rank-2");
        let (rows, cols) = (shape[0], shape[1]);
        let mut m = Self::empty(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let v = src.get(&[i as isize, j as isize]);
                m.set(i as isize, j as isize, v);
            }
        }
        m
    }

    #[inline] pub fn rows(&self) -> usize { self.rows }
    #[inline] pub fn cols(&self) -> usize { self.cols }
    #[inline] pub fn shape(&self) -> [usize; 2] { [self.rows, self.cols] }
    #[inline] pub fn backend(&self) -> &Tensor2D<T> { &self.tensor }
    #[inline] pub fn backend_mut(&mut self) -> &mut Tensor2D<T> { &mut self.tensor }

    /// Dense backend view (compat with older APIs that needed a dense tensor ref).
    #[inline] pub fn dense_backend(&self) -> &DenseTensor<T> { self.tensor.backend() }
    #[inline] pub fn dense_backend_mut(&mut self) -> &mut DenseTensor<T> { self.tensor.backend_mut() }

    #[inline]
    fn assert_compat(&self, rhs: &Self) {
        assert_eq!(self.rows, rhs.rows, "Matrix row mismatch: {} vs {}", self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols, "Matrix col mismatch: {} vs {}", self.cols, rhs.cols);
    }

    #[inline]
    pub fn from_backend(tensor: Tensor2D<T>, rows: usize, cols: usize) -> Self {
        assert_eq!(tensor.shape(), [rows, cols], "Matrix::from_backend: shape mismatch");
        Self { tensor, rows, cols }
    }

    #[inline]
    pub fn cast_to<U: Scalar>(&self) -> Matrix<U> {
        let t_u = self.tensor.cast_to::<U>();
        Matrix::<U>::from_backend(t_u, self.rows, self.cols)
    }

    #[inline]
    pub fn from_ndarray(array: &Array2<T>) -> Self {
        let tensor = Tensor2D::<T>::from_ndarray(array);
        let [rows, cols] = tensor.shape();
        Matrix::from_backend(tensor, rows, cols)
    }

    #[deprecated(note = "use from_ndarray")]
    #[inline]
    pub fn from_ndarry(array: &Array2<T>) -> Self {
        Self::from_ndarray(array)
    }

    #[inline]
    pub fn to_ndarray(&self) -> Array2<T> {
        self.tensor.to_ndarray()
    }

    #[inline]
    pub fn get(&self, i: isize, j: isize) -> T where T: Copy {
        self.tensor.get(i, j)
    }

    #[inline]
    pub fn set(&mut self, i: isize, j: isize, val: T) {
        self.tensor.set(i, j, val)
    }

    #[inline]
    pub fn get_row_ref<'a>(&'a self, i: isize) -> MatrixSlice<'a, T>
    where
        T: Copy,
    {
        MatrixSlice::Borrowed(SliceView { slice: self.tensor.row_view(i) })
    }

    #[inline]
    pub fn get_col_ref<'a>(&'a self, j: isize) -> MatrixSlice<'a, T>
    where
        T: Copy,
    {
        MatrixSlice::Cached(SliceViewCached { buf: self.tensor.col_to_vec(j) })
    }

    #[inline]
    pub fn get_row_ref_mut<'a>(&'a mut self, i: isize) -> MatrixSliceMut<'a, T>
    where
        T: Copy,
    {
        MatrixSliceMut::Borrowed(self.tensor.row_view_mut(i))
    }

    #[inline]
    pub fn get_col_ref_mut<'a>(&'a mut self, j: isize) -> MatrixSliceMut<'a, T>
    where
        T: Copy,
    {
        let c = wrap_axis_index(j, self.cols);
        MatrixSliceMut::Guard(ColMutGuard {
            parent: self as *mut _,
            col: c,
            buf: self.tensor.col_to_vec(j),
            _pd: PhantomData,
        })
    }
}

impl<T: Scalar> MatrixTrait<T> for Matrix<T> {
    #[inline]
    fn rows(&self) -> usize { self.rows }

    #[inline]
    fn cols(&self) -> usize { self.cols }

    #[inline]
    fn get(&self, i: isize, j: isize) -> T { self.tensor.get(i, j) }

    #[inline]
    fn set(&mut self, i: isize, j: isize, val: T) { self.tensor.set(i, j, val) }

    #[inline]
    fn row_view<'a>(&'a self, i: isize) -> &'a [T] { self.tensor.row_view(i) }

    #[inline]
    fn row_view_mut<'a>(&'a mut self, i: isize) -> &'a mut [T] { self.tensor.row_view_mut(i) }

    #[inline]
    fn col_view<'a>(&'a self, _j: isize) -> &'a [T] {
        panic!("col_view: cannot return contiguous column slice in row-major layout")
    }

    #[inline]
    fn col_view_mut<'a>(&'a mut self, _j: isize) -> &'a mut [T] {
        panic!("col_view_mut: cannot return contiguous column slice in row-major layout")
    }

    #[inline]
    fn set_row_from_slice(&mut self, i: isize, vals: &[T]) {
        assert_eq!(vals.len(), self.cols, "set_row_from_slice: len mismatch");
        self.row_view_mut(i).copy_from_slice(vals);
    }

    #[inline]
    fn set_col_from_slice(&mut self, j: isize, vals: &[T]) {
        self.tensor.set_col_from_slice(j, vals)
    }

    #[inline] fn print(&self) { self.tensor.print(); }

    #[inline] fn to_string(&self) { self.tensor.to_string(); }

    #[inline]
    fn transpose(&mut self)
    where
        T: Copy,
    {
        let (rows, cols) = (self.rows, self.cols);
        let mut new_tensor = Tensor2D::<T>::empty(cols, rows);
        for i in 0..rows {
            for j in 0..cols {
                let v = self.get(i as isize, j as isize);
                new_tensor.set(j as isize, i as isize, v);
            }
        }
        self.tensor = new_tensor;
        self.rows = cols;
        self.cols = rows;
    }

    #[inline]
    fn clamp(&mut self, min_val: T, max_val: T)
    where
        T: Scalar + PartialOrd,
    {
        debug_assert!(min_val <= max_val, "clip: min_val must be <= max_val");
        self.tensor.backend_mut().par_map_in_place(|x| {
            let mut y = x;
            if y < min_val { y = min_val; }
            if y > max_val { y = max_val; }
            y
        });
    }

    #[inline]
    fn normalize(&mut self)
    where
        T: Copy + Send + Sync + PartialEq
          + core::ops::Add<Output = T>
          + core::ops::Mul<Output = T>
          + core::ops::Div<Output = T>,
    {
        let mut sum = T::zero();
        for &x in self.tensor.data().iter() { sum = sum + x * x; }
        let n = <T>::sqrt(sum);
        if n == T::zero() { return; }
        self.tensor.backend_mut().par_map_in_place(|x| x / n);
    }

    #[inline]
    fn normalize_by_max(&mut self)
    where
        T: Scalar + PartialOrd
          + core::ops::Mul<Output = T>
          + core::ops::Div<Output = T>,
    {
        let mut max_sq = T::zero();
        for &x in self.tensor.data().iter() {
            let xsq = x * x;
            if xsq > max_sq { max_sq = xsq; }
        }
        let m = <T>::sqrt(max_sq);
        if m == T::zero() { return; }
        self.tensor.backend_mut().par_map_in_place(|x| x / m);
    }
}

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
                let out_dense = &self.tensor.backend().clone() $op rhs.tensor.backend();
                let out = Tensor2D::<T>::from_backend(out_dense, self.rows, self.cols);
                Matrix::<T>::from_backend(out, self.rows, self.cols)
            }
        }
    };
}
impl_matrix_ref_binop!(Add, add, +);
impl_matrix_ref_binop!(Sub, sub, -);
impl_matrix_ref_binop!(Mul, mul, *);
impl_matrix_ref_binop!(Div, div, /);

macro_rules! impl_matrix_ref_assign {
    ($trait:ident, $method:ident) => {
        impl<'a, T> core::ops::$trait<&'a Matrix<T>> for Matrix<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<T>,
        {
            #[inline]
            fn $method(&mut self, rhs: &'a Matrix<T>) {
                self.assert_compat(rhs);
                self.tensor.backend_mut().$method(rhs.tensor.backend());
            }
        }
    };
}
impl_matrix_ref_assign!(AddAssign, add_assign);
impl_matrix_ref_assign!(SubAssign, sub_assign);
impl_matrix_ref_assign!(MulAssign, mul_assign);
impl_matrix_ref_assign!(DivAssign, div_assign);

macro_rules! impl_matrix_ref_scalar_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> core::ops::$trait<T> for &'a Matrix<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<Output = T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                let out_dense = &self.tensor.backend().clone() $op rhs;
                let out = Tensor2D::<T>::from_backend(out_dense, self.rows, self.cols);
                Matrix::<T>::from_backend(out, self.rows, self.cols)
            }
        }
    };
}
impl_matrix_ref_scalar_binop!(Add, add, +);
impl_matrix_ref_scalar_binop!(Sub, sub, -);
impl_matrix_ref_scalar_binop!(Mul, mul, *);
impl_matrix_ref_scalar_binop!(Div, div, /);

macro_rules! impl_matrix_scalar_assign {
    ($trait:ident, $method:ident) => {
        impl<T> core::ops::$trait<T> for Matrix<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<T>,
        {
            #[inline]
            fn $method(&mut self, rhs: T) {
                self.tensor.backend_mut().$method(rhs);
            }
        }
    };
}
impl_matrix_scalar_assign!(AddAssign, add_assign);
impl_matrix_scalar_assign!(SubAssign, sub_assign);
impl_matrix_scalar_assign!(MulAssign, mul_assign);
impl_matrix_scalar_assign!(DivAssign, div_assign);

impl<T: Scalar> NdarrayConvert for Matrix<T> {
    type NdArray = Array2<T>;

    #[inline]
    fn from_ndarray(array: &Self::NdArray) -> Self {
        Matrix::<T>::from_ndarray(array)
    }

    #[inline]
    fn to_ndarray(&self) -> Self::NdArray {
        Matrix::<T>::to_ndarray(self)
    }
}

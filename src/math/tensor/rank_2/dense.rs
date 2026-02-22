/*!
Core dense 2D tensor with fixed row-major physical layout.
*/

use ndarray::Array2;

use crate::math::{
    ndarray_convert::NdarrayConvert,
    scalar::Scalar,
    tensor::core::{dense::Tensor as DenseTensor, tensor_trait::TensorTrait},
};

#[derive(Debug, Clone)]
pub struct Tensor2D<T: Scalar> {
    rows: usize,
    cols: usize,
    backend: DenseTensor<T>, // shape [rows, cols], row-major
}

#[inline(always)]
fn wrap_axis_index(idx: isize, dim: usize) -> usize {
    debug_assert!(dim > 0);
    let d = dim as isize;
    let mut m = idx % d;
    if m < 0 { m += d; }
    m as usize
}

impl<T: Scalar> Tensor2D<T> {
    #[inline]
    pub fn empty(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "Tensor2D::empty: shape must be nonzero");
        let backend = DenseTensor::<T>::empty(&[rows, cols]);
        Self { rows, cols, backend }
    }

    #[inline]
    pub fn from_backend(backend: DenseTensor<T>, rows: usize, cols: usize) -> Self {
        assert_eq!(backend.shape.as_slice(), &[rows, cols], "Tensor2D::from_backend: shape mismatch");
        Self { rows, cols, backend }
    }

    #[inline]
    pub fn rows(&self) -> usize { self.rows }

    #[inline]
    pub fn cols(&self) -> usize { self.cols }

    #[inline]
    pub fn shape(&self) -> [usize; 2] { [self.rows, self.cols] }

    #[inline]
    pub fn backend(&self) -> &DenseTensor<T> { &self.backend }

    #[inline]
    pub fn backend_mut(&mut self) -> &mut DenseTensor<T> { &mut self.backend }

    #[inline]
    pub fn data(&self) -> &[T] { &self.backend.data }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] { &mut self.backend.data }

    #[inline]
    pub fn get(&self, i: isize, j: isize) -> T where T: Copy {
        self.backend.get(&[i, j])
    }

    #[inline]
    pub fn set(&mut self, i: isize, j: isize, val: T) {
        self.backend.set(&[i, j], val);
    }

    #[inline]
    pub fn row_view<'a>(&'a self, i: isize) -> &'a [T] {
        let r = wrap_axis_index(i, self.rows);
        let off = r * self.cols;
        &self.backend.data[off .. off + self.cols]
    }

    #[inline]
    pub fn row_view_mut<'a>(&'a mut self, i: isize) -> &'a mut [T] {
        let r = wrap_axis_index(i, self.rows);
        let off = r * self.cols;
        &mut self.backend.data[off .. off + self.cols]
    }

    #[inline]
    pub fn col_to_vec(&self, j: isize) -> Vec<T>
    where
        T: Copy,
    {
        let c = wrap_axis_index(j, self.cols);
        (0..self.rows)
            .map(|i| self.backend.get(&[i as isize, c as isize]))
            .collect()
    }

    #[inline]
    pub fn set_col_from_slice(&mut self, j: isize, vals: &[T]) {
        assert_eq!(vals.len(), self.rows, "Tensor2D::set_col_from_slice: len mismatch");
        let c = wrap_axis_index(j, self.cols);
        for (i, &v) in vals.iter().enumerate() {
            self.backend.set(&[i as isize, c as isize], v);
        }
    }

    #[inline]
    pub fn cast_to<U: Scalar>(&self) -> Tensor2D<U> {
        let b = self.backend.cast_to::<U>();
        Tensor2D::<U>::from_backend(b, self.rows, self.cols)
    }

    #[inline]
    pub fn from_ndarray(array: &Array2<T>) -> Self {
        let shape = array.shape();
        let (rows, cols) = (shape[0], shape[1]);
        let owned = array.to_owned();
        let (data, _) = owned.into_raw_vec_and_offset();
        let backend = DenseTensor::<T> { shape: vec![rows, cols], data };
        Self::from_backend(backend, rows, cols)
    }

    #[inline]
    pub fn to_ndarray(&self) -> Array2<T> {
        Array2::from_shape_vec((self.rows, self.cols), self.backend.data.clone())
            .expect("Tensor2D::to_ndarray: shape/data length mismatch")
    }

    #[inline]
    pub fn print(&self) { self.backend.print(); }

    #[inline]
    pub fn to_string(&self) { self.backend.to_string(); }
}

impl<T: Scalar> NdarrayConvert for Tensor2D<T> {
    type NdArray = Array2<T>;

    #[inline]
    fn from_ndarray(array: &Self::NdArray) -> Self {
        Tensor2D::<T>::from_ndarray(array)
    }

    #[inline]
    fn to_ndarray(&self) -> Self::NdArray {
        Tensor2D::<T>::to_ndarray(self)
    }
}

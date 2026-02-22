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
/// Annotation:
/// - Purpose: Executes `wrap_axis_index` logic for this module.
/// - Parameters:
///   - `idx` (`isize`): Index argument selecting an element or slot.
///   - `dim` (`usize`): Parameter of type `usize` used by `wrap_axis_index`.
fn wrap_axis_index(idx: isize, dim: usize) -> usize {
    debug_assert!(dim > 0);
    let d = dim as isize;
    let mut m = idx % d;
    if m < 0 { m += d; }
    m as usize
}

impl<T: Scalar> Tensor2D<T> {
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `empty` logic for this module.
    /// - Parameters:
    ///   - `rows` (`usize`): Parameter of type `usize` used by `empty`.
    ///   - `cols` (`usize`): Parameter of type `usize` used by `empty`.
    pub fn empty(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "Tensor2D::empty: shape must be nonzero");
        let backend = DenseTensor::<T>::empty(&[rows, cols]);
        Self { rows, cols, backend }
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `backend` input.
    /// - Parameters:
    ///   - `backend` (`DenseTensor<T>`): Parameter of type `DenseTensor<T>` used by `from_backend`.
    ///   - `rows` (`usize`): Parameter of type `usize` used by `from_backend`.
    ///   - `cols` (`usize`): Parameter of type `usize` used by `from_backend`.
    pub fn from_backend(backend: DenseTensor<T>, rows: usize, cols: usize) -> Self {
        assert_eq!(backend.shape.as_slice(), &[rows, cols], "Tensor2D::from_backend: shape mismatch");
        Self { rows, cols, backend }
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `rows` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn rows(&self) -> usize { self.rows }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `cols` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn cols(&self) -> usize { self.cols }

    #[inline]
    /// Annotation:
    /// - Purpose: Returns the logical shape metadata.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn shape(&self) -> [usize; 2] { [self.rows, self.cols] }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `backend` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn backend(&self) -> &DenseTensor<T> { &self.backend }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `backend_mut` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn backend_mut(&mut self) -> &mut DenseTensor<T> { &mut self.backend }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `data` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn data(&self) -> &[T] { &self.backend.data }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `data_mut` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn data_mut(&mut self) -> &mut [T] { &mut self.backend.data }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `get` logic for this module.
    /// - Parameters:
    ///   - `i` (`isize`): Primary index argument.
    ///   - `j` (`isize`): Secondary index argument.
    pub fn get(&self, i: isize, j: isize) -> T where T: Copy {
        self.backend.get(&[i, j])
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `set` logic for this module.
    /// - Parameters:
    ///   - `i` (`isize`): Primary index argument.
    ///   - `j` (`isize`): Secondary index argument.
    ///   - `val` (`T`): Value provided by caller for write/update behavior.
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
    /// Annotation:
    /// - Purpose: Executes `col_to_vec` logic for this module.
    /// - Parameters:
    ///   - `j` (`isize`): Secondary index argument.
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
    /// Annotation:
    /// - Purpose: Sets the `col_from_slice` value.
    /// - Parameters:
    ///   - `j` (`isize`): Secondary index argument.
    ///   - `vals` (`&[T]`): Parameter of type `&[T]` used by `set_col_from_slice`.
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
    /// Annotation:
    /// - Purpose: Builds this value from `ndarray` input.
    /// - Parameters:
    ///   - `array` (`&Array2<T>`): ndarray input used for conversion/interoperability.
    pub fn from_ndarray(array: &Array2<T>) -> Self {
        let shape = array.shape();
        let (rows, cols) = (shape[0], shape[1]);
        let owned = array.to_owned();
        let (data, _) = owned.into_raw_vec_and_offset();
        let backend = DenseTensor::<T> { shape: vec![rows, cols], data };
        Self::from_backend(backend, rows, cols)
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_ndarray(&self) -> Array2<T> {
        Array2::from_shape_vec((self.rows, self.cols), self.backend.data.clone())
            .expect("Tensor2D::to_ndarray: shape/data length mismatch")
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Prints a human-readable representation.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn print(&self) { self.backend.print(); }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `string` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn to_string(&self) { self.backend.to_string(); }
}

impl<T: Scalar> NdarrayConvert for Tensor2D<T> {
    type NdArray = Array2<T>;

    #[inline]
    /// Annotation:
    /// - Purpose: Builds this value from `ndarray` input.
    /// - Parameters:
    ///   - `array` (`&Self::NdArray`): ndarray input used for conversion/interoperability.
    fn from_ndarray(array: &Self::NdArray) -> Self {
        Tensor2D::<T>::from_ndarray(array)
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Converts this value into `ndarray` form.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    fn to_ndarray(&self) -> Self::NdArray {
        Tensor2D::<T>::to_ndarray(self)
    }
}

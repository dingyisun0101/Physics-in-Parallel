/*!
A container for many fixed-length vectors with external logical shape `[dim, n]`.

Internal storage
----------------
- Uses core `tensor::rank_2::dense::Tensor2D` in **row-major** layout.
- Physical shape is `[n, dim]` so each vector is contiguous in memory.
*/

pub mod rand;

use ndarray::Array2;
use rayon::prelude::*;

use crate::math::{
    ndarray_convert::NdarrayConvert,
    scalar::Scalar,
    tensor::{
        core::dense::Tensor as DenseTensor,
        core::tensor_trait::TensorTrait,
        rank_2::dense::Tensor2D,
    },
};

#[derive(Debug, Clone)]
pub struct VectorList<T: Scalar> {
    /// Backing tensor with physical shape `[n, dim]` (row-major).
    pub tensor: Tensor2D<T>,
}

impl<T: Scalar> VectorList<T> {
    #[inline]
    pub fn empty(dim: usize, n: usize) -> Self {
        assert!(dim > 0 && n > 0, "VectorList::empty: shape must be nonzero");
        Self { tensor: Tensor2D::<T>::empty(n, dim) }
    }

    #[inline]
    pub fn from_tensor2d(tensor: Tensor2D<T>) -> Self {
        Self { tensor }
    }

    #[inline]
    pub fn into_tensor2d(self) -> Tensor2D<T> { self.tensor }

    /// Dense backend view (compat layer for fillers that operate on dense tensors).
    #[inline]
    pub fn as_tensor(&self) -> &DenseTensor<T> { self.tensor.backend() }

    /// Dense backend mutable view (compat layer for fillers that operate on dense tensors).
    #[inline]
    pub fn as_tensor_mut(&mut self) -> &mut DenseTensor<T> { self.tensor.backend_mut() }

    // ----------------------- shape helpers -----------------------
    /// Vector dimension `dim`.
    #[inline] pub fn dim(&self) -> usize { self.tensor.cols() }
    /// Number of vectors `n`.
    #[inline] pub fn num_vectors(&self) -> usize { self.tensor.rows() }
    /// External logical shape `[dim, n]`.
    #[inline] pub fn shape(&self) -> [usize; 2] { [self.dim(), self.num_vectors()] }

    // ------------------------ I/O helpers ------------------------
    #[inline]
    pub fn print(&self) { self.tensor.print(); }

    #[inline]
    pub fn cast_to<U>(&self) -> VectorList<U>
    where
        U: Scalar,
    {
        VectorList { tensor: self.tensor.cast_to::<U>() }
    }

    /// Build from ndarray with shape `[dim, n]`.
    #[inline]
    pub fn from_ndarray(array: &Array2<T>) -> Self {
        let shape = array.shape();
        assert!(shape[0] > 0 && shape[1] > 0, "VectorList::from_ndarray: shape must be nonzero");

        let dim = shape[0];
        let n = shape[1];
        let mut out = Tensor2D::<T>::empty(n, dim);
        for i in 0..n {
            for k in 0..dim {
                out.set(i as isize, k as isize, array[(k, i)]);
            }
        }
        Self { tensor: out }
    }

    /// Convert to ndarray with shape `[dim, n]`.
    #[inline]
    pub fn to_ndarray(&self) -> Array2<T> {
        let (dim, n) = (self.dim(), self.num_vectors());
        let mut data = Vec::with_capacity(dim * n);
        for k in 0..dim {
            for i in 0..n {
                data.push(self.tensor.get(i as isize, k as isize));
            }
        }
        Array2::from_shape_vec((dim, n), data)
            .expect("VectorList::to_ndarray: shape/data length mismatch")
    }

    // --------------------- element accessors ---------------------
    #[inline]
    pub fn get(&self, i: isize, k: isize) -> T where T: Copy {
        self.tensor.get(i, k)
    }

    #[inline]
    pub fn set(&mut self, i: isize, k: isize, val: T) {
        self.tensor.set(i, k, val);
    }

    // ------------------------- views -----------------------------

    /// Zero-copy vector view (length = `dim`).
    #[inline]
    pub fn get_vector<'a>(&'a self, i: isize) -> &'a [T]
    where
        T: Copy,
    {
        self.tensor.row_view(i)
    }

    /// Mutable zero-copy vector view (length = `dim`).
    #[inline]
    pub fn get_vector_mut<'a>(&'a mut self, i: isize) -> &'a mut [T]
    where
        T: Copy,
    {
        self.tensor.row_view_mut(i)
    }

    /// Copy one axis (component) across all vectors.
    #[inline]
    pub fn get_axis(&self, k: isize) -> Vec<T>
    where
        T: Copy,
    {
        self.tensor.col_to_vec(k)
    }
}

impl<T: Scalar> VectorList<T> {
    #[inline]
    pub fn scale_vectors_by_list(&mut self, scales: &[T]) {
        let n = self.num_vectors();
        assert!(scales.len() == n, "scale_vectors_by_list: len mismatch");

        for (i, &s) in scales.iter().enumerate() {
            for x in self.tensor.row_view_mut(i as isize) {
                *x = *x * s;
            }
        }
    }

    #[inline]
    pub fn set_vector_from_slice(&mut self, i: isize, vals: &[T]) {
        assert_eq!(vals.len(), self.dim(), "set_vector_from_slice: len mismatch");
        self.tensor.row_view_mut(i).copy_from_slice(vals);
    }

    #[inline]
    pub fn set_axis_from_slice(&mut self, k: isize, vals: &[T]) {
        assert_eq!(vals.len(), self.num_vectors(), "set_axis_from_slice: len mismatch");
        self.tensor.set_col_from_slice(k, vals);
    }

    #[inline]
    pub fn fill(&mut self, val: T) {
        self.tensor.backend_mut().par_fill(val);
    }

    #[inline]
    pub fn get_norms(&self) -> Vec<T> {
        (0..self.num_vectors())
            .into_par_iter()
            .map(|i| {
                let mut s = T::zero();
                for &x in self.tensor.row_view(i as isize) {
                    s = s + x * x;
                }
                <T>::sqrt(s)
            })
            .collect()
    }

    #[inline]
    pub fn normalize(&mut self) {
        let norms = self.get_norms();
        let scales: Vec<T> = norms.par_iter()
            .map(|&n| if n == T::zero() { T::one() } else { T::one() / n })
            .collect();
        self.scale_vectors_by_list(&scales);
    }

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

macro_rules! impl_vl_ref_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> core::ops::$trait<&'a VectorList<T>> for &'a VectorList<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<Output = T>,
        {
            type Output = VectorList<T>;
            #[inline]
            fn $method(self, rhs: &'a VectorList<T>) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape(), "VectorList shape mismatch");
                let out_dense = &self.tensor.backend().clone() $op rhs.tensor.backend();
                let out = Tensor2D::<T>::from_backend(out_dense, self.num_vectors(), self.dim());
                VectorList { tensor: out }
            }
        }
    };
}
impl_vl_ref_binop!(Add, add, +);
impl_vl_ref_binop!(Sub, sub, -);
impl_vl_ref_binop!(Mul, mul, *);
impl_vl_ref_binop!(Div, div, /);

macro_rules! impl_vl_ref_assign {
    ($trait:ident, $method:ident) => {
        impl<'a, T> core::ops::$trait<&'a VectorList<T>> for VectorList<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<T>,
        {
            #[inline]
            fn $method(&mut self, rhs: &'a VectorList<T>) {
                assert_eq!(self.shape(), rhs.shape(), "VectorList shape mismatch");
                self.tensor.backend_mut().$method(rhs.tensor.backend());
            }
        }
    };
}
impl_vl_ref_assign!(AddAssign, add_assign);
impl_vl_ref_assign!(SubAssign, sub_assign);
impl_vl_ref_assign!(MulAssign, mul_assign);
impl_vl_ref_assign!(DivAssign, div_assign);

macro_rules! impl_vl_ref_scalar_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> core::ops::$trait<T> for &'a VectorList<T>
        where
            T: Scalar + Copy + Send + Sync + core::ops::$trait<Output = T>,
        {
            type Output = VectorList<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                let out_dense = &self.tensor.backend().clone() $op rhs;
                let out = Tensor2D::<T>::from_backend(out_dense, self.num_vectors(), self.dim());
                VectorList { tensor: out }
            }
        }
    };
}
impl_vl_ref_scalar_binop!(Add, add, +);
impl_vl_ref_scalar_binop!(Sub, sub, -);
impl_vl_ref_scalar_binop!(Mul, mul, *);
impl_vl_ref_scalar_binop!(Div, div, /);

macro_rules! impl_vl_scalar_assign {
    ($trait:ident, $method:ident) => {
        impl<T> core::ops::$trait<T> for VectorList<T>
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
impl_vl_scalar_assign!(AddAssign, add_assign);
impl_vl_scalar_assign!(SubAssign, sub_assign);
impl_vl_scalar_assign!(MulAssign, mul_assign);
impl_vl_scalar_assign!(DivAssign, div_assign);

impl<T: Scalar> NdarrayConvert for VectorList<T> {
    type NdArray = Array2<T>;

    #[inline]
    fn from_ndarray(array: &Self::NdArray) -> Self {
        VectorList::<T>::from_ndarray(array)
    }

    #[inline]
    fn to_ndarray(&self) -> Self::NdArray {
        VectorList::<T>::to_ndarray(self)
    }
}

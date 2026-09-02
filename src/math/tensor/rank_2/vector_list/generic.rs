/*!
Vector-list wrapper over dense rank-N tensor storage.

Purpose:
    `VectorList<T>` stores a long list of fixed-length vectors as one dense
    rank-N tensor with logical shape `[num_vectors, dim]`. The wrapper exists so
    users can work with complete vectors as the common manipulation unit instead
    of repeatedly addressing individual scalar tensor entries.

Storage model:
    - Backend:
        `Tensor<T, Dense>`.
    - Shape:
        `[num_vectors, dim]`, where `num_vectors` is the number of vectors and `dim`
        is the component count in each vector.
    - Layout:
        Row-major dense storage. Each vector occupies one contiguous row, so
        `vector` and `vector_mut` can return zero-copy slices.

API boundary:
    - Public methods use vector-list language: `vector`, `set_vector`,
        `num_vectors`, `dim`, `axis`, `normalize`.
    - Scalar `get` and `set` remain available for sanity checks and targeted
        component edits.
    - Direct dense tensor access is crate-internal. Other PiP modules may use it
        for fast fillers and IO, but end users do not need to reason about the
        backend.

Parallelism:
    Operations that touch every vector use Rayon over row chunks. This is the
    central performance reason for keeping vectors contiguous in dense storage:
    Haar-vector normalization, nearest-neighbor decoding, and bulk vector
    transforms can run independently over many rows.
*/

use std::any::Any;

use crate::math::tensor::rank_n::dense::Tensor as DenseStorage;
use crate::math::tensor::rank_n::{Dense, Tensor};
use crate::math::{Scalar, ScalarCastError};
use crate::threading::parallel_chunk_len;
use num_traits::Zero;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct VectorList<T: Scalar> {
    tensor: Tensor<T, Dense>,
}

pub trait DynVectorList: std::fmt::Debug + Send + Sync + erased_serde::Serialize {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn dim(&self) -> usize;
    fn num_vectors(&self) -> usize;
    fn type_name(&self) -> &'static str;
    fn scalar_kind(&self) -> &'static str;
    fn clone_box(&self) -> Box<dyn DynVectorList>;
}

erased_serde::serialize_trait_object!(DynVectorList);

impl Clone for Box<dyn DynVectorList> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl<T: Scalar> VectorList<T> {
    #[inline]
    pub fn empty(dim: usize, num_vectors: usize) -> Self {
        assert!(
            dim > 0 && num_vectors > 0,
            "VectorList::empty: dim and num_vectors must be nonzero"
        );
        Self {
            tensor: Tensor::<T, Dense>::empty(&[num_vectors, dim]),
        }
    }

    #[inline]
    pub fn zeros(dim: usize, num_vectors: usize) -> Self {
        Self::empty(dim, num_vectors)
    }

    #[inline]
    pub fn from_vec(dim: usize, num_vectors: usize, data: Vec<T>) -> Self {
        assert!(
            dim > 0 && num_vectors > 0,
            "VectorList::from_vec: dim and num_vectors must be nonzero"
        );
        Self {
            tensor: Tensor::<T, Dense>::from_vec(&[num_vectors, dim], data),
        }
    }

    pub fn from_fn<F>(dim: usize, num_vectors: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let data = (0..num_vectors)
            .flat_map(|i| (0..dim).map(move |k| (i, k)))
            .map(|(i, k)| f(i, k))
            .collect();
        Self::from_vec(dim, num_vectors, data)
    }

    #[inline]
    pub(crate) fn from_tensor(tensor: Tensor<T, Dense>) -> Self {
        assert_eq!(
            tensor.rank(),
            2,
            "VectorList::from_tensor: dense tensor must have rank 2"
        );
        assert!(
            tensor.shape()[0] > 0 && tensor.shape()[1] > 0,
            "VectorList::from_tensor: shape must be nonzero"
        );
        Self { tensor }
    }

    #[inline]
    pub(crate) fn as_tensor(&self) -> &DenseStorage<T> {
        self.tensor.storage()
    }

    #[inline]
    pub(crate) fn as_tensor_mut(&mut self) -> &mut DenseStorage<T> {
        self.tensor.storage_mut()
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.tensor.shape()[1]
    }

    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.tensor.shape()[0]
    }

    #[inline]
    pub fn shape(&self) -> [usize; 2] {
        [self.num_vectors(), self.dim()]
    }

    #[inline]
    pub fn get(&self, i: isize, k: isize) -> T
    where
        T: Copy,
    {
        self.tensor.get(&[i, k])
    }

    #[inline]
    pub fn set(&mut self, i: isize, k: isize, value: T) {
        self.tensor.set(&[i, k], value);
    }

    #[inline]
    pub fn vector(&self, i: isize) -> &[T] {
        let row = wrap_index(i, self.num_vectors());
        let dim = self.dim();
        let start = row * dim;
        &self.as_tensor().data()[start..start + dim]
    }

    #[inline]
    pub fn vector_mut(&mut self, i: isize) -> &mut [T] {
        let row = wrap_index(i, self.num_vectors());
        let dim = self.dim();
        let start = row * dim;
        &mut self.as_tensor_mut().data_mut()[start..start + dim]
    }

    #[inline]
    pub fn vector_owned(&self, i: isize) -> Vec<T>
    where
        T: Copy,
    {
        self.vector(i).to_vec()
    }

    #[inline]
    pub fn set_vector(&mut self, i: isize, values: &[T])
    where
        T: Copy,
    {
        assert_eq!(
            values.len(),
            self.dim(),
            "VectorList::set_vector: vector length mismatch"
        );
        self.vector_mut(i).copy_from_slice(values);
    }

    #[inline]
    pub fn axis(&self, k: isize) -> Vec<T>
    where
        T: Copy,
    {
        let axis = wrap_index(k, self.dim());
        self.as_tensor()
            .data()
            .chunks_exact(self.dim())
            .map(|row| row[axis])
            .collect()
    }

    #[inline]
    pub fn set_axis(&mut self, k: isize, values: &[T])
    where
        T: Copy,
    {
        assert_eq!(
            values.len(),
            self.num_vectors(),
            "VectorList::set_axis: length mismatch"
        );
        let axis = wrap_index(k, self.dim());
        let dim = self.dim();
        self.as_tensor_mut()
            .data_mut()
            .chunks_exact_mut(dim)
            .zip(values.iter().copied())
            .for_each(|(row, value)| row[axis] = value);
    }

    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.tensor.fill(value);
    }

    pub fn par_for_each_vector<F>(&self, f: F)
    where
        T: Sync,
        F: Fn(usize, &[T]) + Send + Sync,
    {
        self.as_tensor()
            .data()
            .par_chunks_exact(self.dim())
            .with_min_len(parallel_chunk_len(self.num_vectors()).unwrap_or(1))
            .enumerate()
            .for_each(|(i, row)| f(i, row));
    }

    pub fn par_for_each_vector_mut<F>(&mut self, f: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Send + Sync,
    {
        let dim = self.dim();
        let min_vectors_per_job = parallel_chunk_len(self.num_vectors()).unwrap_or(1);
        self.as_tensor_mut()
            .data_mut()
            .par_chunks_exact_mut(dim)
            .with_min_len(min_vectors_per_job)
            .enumerate()
            .for_each(|(i, row)| f(i, row));
    }

    pub fn scale_vectors_by_list(&mut self, scales: &[T])
    where
        T: Copy + Send + Sync,
    {
        assert_eq!(
            scales.len(),
            self.num_vectors(),
            "VectorList::scale_vectors_by_list: length mismatch"
        );
        self.par_for_each_vector_mut(|i, row| {
            let scale = scales[i];
            row.iter_mut().for_each(|x| *x = *x * scale);
        });
    }

    pub fn norms(&self) -> Vec<T>
    where
        T: Copy + Send + Sync,
    {
        self.as_tensor()
            .data()
            .par_chunks_exact(self.dim())
            .with_min_len(parallel_chunk_len(self.num_vectors()).unwrap_or(1))
            .map(|row| {
                let sum = row
                    .iter()
                    .copied()
                    .map(Scalar::norm_sqr)
                    .reduce(|a, b| a + b)
                    .unwrap_or_else(T::zero);
                sum.sqrt()
            })
            .collect()
    }

    pub fn norms_real(&self) -> Vec<T::Real>
    where
        T: Copy + Send + Sync,
        T::Real: Send + Sync,
    {
        self.as_tensor()
            .data()
            .par_chunks_exact(self.dim())
            .with_min_len(parallel_chunk_len(self.num_vectors()).unwrap_or(1))
            .map(|row| {
                row.iter()
                    .copied()
                    .map(|x| x.norm_sqr_real())
                    .reduce(|a, b| a + b)
                    .unwrap_or_else(T::Real::zero)
                    .sqrt()
            })
            .collect()
    }

    pub fn normalize(&mut self)
    where
        T: Copy + Send + Sync,
    {
        let norms = self.norms();
        let scales: Vec<T> = norms
            .par_iter()
            .with_min_len(parallel_chunk_len(norms.len()).unwrap_or(1))
            .copied()
            .map(|norm| {
                if norm == T::zero() {
                    T::one()
                } else {
                    T::one() / norm
                }
            })
            .collect();
        self.scale_vectors_by_list(&scales);
    }

    pub fn to_polar(&self) -> (Vec<T>, Self)
    where
        T: Copy + Send + Sync,
    {
        let norms = self.norms();
        let mut units = self.clone();
        let scales: Vec<T> = norms
            .par_iter()
            .with_min_len(parallel_chunk_len(norms.len()).unwrap_or(1))
            .copied()
            .map(|norm| {
                if norm == T::zero() {
                    T::zero()
                } else {
                    T::one() / norm
                }
            })
            .collect();
        units.scale_vectors_by_list(&scales);
        (norms, units)
    }

    #[inline]
    pub fn add(&self, rhs: &Self) -> Self
    where
        T: Copy + Send + Sync,
    {
        assert_eq!(self.shape(), rhs.shape(), "VectorList shape mismatch");
        Self::from_tensor(self.tensor.zip_with(&rhs.tensor, |a, b| a + b))
    }

    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self
    where
        T: Copy + Send + Sync,
    {
        assert_eq!(self.shape(), rhs.shape(), "VectorList shape mismatch");
        Self::from_tensor(self.tensor.zip_with(&rhs.tensor, |a, b| a - b))
    }

    #[inline]
    pub fn elem_mul(&self, rhs: &Self) -> Self
    where
        T: Copy + Send + Sync,
    {
        assert_eq!(self.shape(), rhs.shape(), "VectorList shape mismatch");
        Self::from_tensor(self.tensor.elem_mul(&rhs.tensor))
    }

    #[inline]
    pub fn elem_div(&self, rhs: &Self) -> Self
    where
        T: Copy + Send + Sync,
    {
        assert_eq!(self.shape(), rhs.shape(), "VectorList shape mismatch");
        Self::from_tensor(self.tensor.elem_div(&rhs.tensor))
    }

    #[inline]
    pub fn scalar_mul(&self, scalar: T) -> Self
    where
        T: Copy + Send + Sync,
    {
        Self::from_tensor(self.tensor.scalar_mul(scalar))
    }

    #[inline]
    pub fn try_cast_to<U: Scalar>(&self) -> Result<VectorList<U>, ScalarCastError> {
        self.tensor.try_cast_to::<U>().map(VectorList::from_tensor)
    }

    #[inline]
    pub fn cast_to<U: Scalar + Send + Sync>(&self) -> VectorList<U> {
        VectorList::from_tensor(self.tensor.cast_to::<U>())
    }
}

macro_rules! impl_vl_ref_binop {
    ($trait:ident, $method:ident, $op_method:ident) => {
        impl<'a, T> core::ops::$trait<&'a VectorList<T>> for &'a VectorList<T>
        where
            T: Scalar + Copy + Send + Sync,
        {
            type Output = VectorList<T>;

            #[inline]
            fn $method(self, rhs: &'a VectorList<T>) -> Self::Output {
                self.$op_method(rhs)
            }
        }
    };
}

impl_vl_ref_binop!(Add, add, add);
impl_vl_ref_binop!(Sub, sub, sub);
impl_vl_ref_binop!(Mul, mul, elem_mul);
impl_vl_ref_binop!(Div, div, elem_div);

macro_rules! impl_vl_scalar_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<'a, T> core::ops::$trait<T> for &'a VectorList<T>
        where
            T: Scalar + Copy + Send + Sync,
        {
            type Output = VectorList<T>;

            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                let data: Vec<T> = self
                    .as_tensor()
                    .data()
                    .par_iter()
                    .with_min_len(parallel_chunk_len(self.as_tensor().data().len()).unwrap_or(1))
                    .copied()
                    .map(|x| x $op rhs)
                    .collect();
                VectorList::from_vec(self.dim(), self.num_vectors(), data)
            }
        }
    };
}

impl_vl_scalar_binop!(Add, add, +);
impl_vl_scalar_binop!(Sub, sub, -);
impl_vl_scalar_binop!(Mul, mul, *);
impl_vl_scalar_binop!(Div, div, /);

#[inline]
fn wrap_index(index: isize, len: usize) -> usize {
    debug_assert!(len > 0);
    let len = len as isize;
    let mut wrapped = index % len;
    if wrapped < 0 {
        wrapped += len;
    }
    wrapped as usize
}

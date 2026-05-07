/*!
Vector-list wrapper over dense rank-N tensor storage.

Purpose:
    `VectorList<T>` stores a long list of fixed-length vectors as one dense
    rank-N tensor with logical shape `[num_vecs, dim]`. The wrapper exists so
    users can work with complete vectors as the common manipulation unit instead
    of repeatedly addressing individual scalar tensor entries.

Storage model:
    - Backend:
        `Tensor<T, Dense>`.
    - Shape:
        `[num_vecs, dim]`, where `num_vecs` is the number of vectors and `dim`
        is the component count in each vector.
    - Layout:
        Row-major dense storage. Each vector occupies one contiguous row, so
        `get_vec` and `get_vec_mut` can return zero-copy slices.

API boundary:
    - Public methods use vector-list language: `get_vec`, `set_vec`,
        `num_vecs`, `dim`, `axis`, `normalize`.
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

use ndarray::Array2;
use num_traits::Zero;
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::math::io::json::{FlatPayload, FromJsonPayload, ToJsonPayload};
use crate::math::tensor::rank_n::dense::Tensor as DenseStorage;
use crate::math::tensor::{Dense, Tensor};
use crate::math::{NdarrayConvert, Scalar, ScalarCastError};

#[derive(Debug, Clone)]
pub struct VectorList<T: Scalar> {
    tensor: Tensor<T, Dense>,
}

pub trait DynVectorList: std::fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn dim(&self) -> usize;
    fn num_vecs(&self) -> usize;
    fn type_name(&self) -> &'static str;
    fn clone_box(&self) -> Box<dyn DynVectorList>;
    fn serialize_value(&self) -> Result<Value, serde_json::Error>;
    fn serialize(&self) -> Result<String, serde_json::Error>;
}

impl Clone for Box<dyn DynVectorList> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl<T> DynVectorList for VectorList<T>
where
    T: Scalar + Serialize + Copy + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dim(&self) -> usize {
        self.dim()
    }

    fn num_vecs(&self) -> usize {
        self.num_vecs()
    }

    fn type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    fn clone_box(&self) -> Box<dyn DynVectorList> {
        Box::new(self.clone())
    }

    fn serialize_value(&self) -> Result<Value, serde_json::Error> {
        VectorList::<T>::serialize_value(self)
    }

    fn serialize(&self) -> Result<String, serde_json::Error> {
        VectorList::<T>::serialize(self)
    }
}

impl<T> Serialize for VectorList<T>
where
    T: Scalar + Serialize + Copy,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_json_payload()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for VectorList<T>
where
    T: Scalar + DeserializeOwned + Copy,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = FlatPayload::<T>::deserialize(deserializer)?;
        <Self as FromJsonPayload>::from_json_payload(payload).map_err(serde::de::Error::custom)
    }
}

impl<T> ToJsonPayload for VectorList<T>
where
    T: Scalar + Serialize + Copy,
{
    type Payload = FlatPayload<T>;

    fn to_json_payload(&self) -> Result<Self::Payload, serde_json::Error> {
        Ok(FlatPayload::new(
            "vector_list",
            vec![self.num_vecs(), self.dim()],
            self.as_tensor().data().to_vec(),
        ))
    }
}

impl<T> FromJsonPayload for VectorList<T>
where
    T: Scalar + DeserializeOwned + Copy,
{
    type Payload = FlatPayload<T>;

    fn from_json_payload(payload: Self::Payload) -> Result<Self, String> {
        payload.validate_dense("vector_list")?;
        if payload.shape.len() != 2 {
            return Err(format!(
                "vector_list shape rank mismatch: expected 2, got {}",
                payload.shape.len()
            ));
        }
        let num_vecs = payload.shape[0];
        let dim = payload.shape[1];
        Ok(Self::from_vec(dim, num_vecs, payload.data))
    }
}

impl<T: Scalar> VectorList<T> {
    #[inline]
    pub fn empty(dim: usize, num_vecs: usize) -> Self {
        assert!(
            dim > 0 && num_vecs > 0,
            "VectorList::empty: dim and num_vecs must be nonzero"
        );
        Self {
            tensor: Tensor::<T, Dense>::empty(&[num_vecs, dim]),
        }
    }

    #[inline]
    pub fn zeros(dim: usize, num_vecs: usize) -> Self {
        Self::empty(dim, num_vecs)
    }

    #[inline]
    pub fn from_vec(dim: usize, num_vecs: usize, data: Vec<T>) -> Self {
        assert!(
            dim > 0 && num_vecs > 0,
            "VectorList::from_vec: dim and num_vecs must be nonzero"
        );
        Self {
            tensor: Tensor::<T, Dense>::from_vec(&[num_vecs, dim], data),
        }
    }

    pub fn from_fn<F>(dim: usize, num_vecs: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let data = (0..num_vecs)
            .flat_map(|i| (0..dim).map(move |k| (i, k)))
            .map(|(i, k)| f(i, k))
            .collect();
        Self::from_vec(dim, num_vecs, data)
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
    pub fn num_vecs(&self) -> usize {
        self.tensor.shape()[0]
    }

    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.num_vecs()
    }

    #[inline]
    pub fn shape(&self) -> [usize; 2] {
        [self.num_vecs(), self.dim()]
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
    pub fn get_vec(&self, i: isize) -> &[T] {
        let row = wrap_index(i, self.num_vecs());
        let dim = self.dim();
        let start = row * dim;
        &self.as_tensor().data()[start..start + dim]
    }

    #[inline]
    pub fn get_vec_mut(&mut self, i: isize) -> &mut [T] {
        let row = wrap_index(i, self.num_vecs());
        let dim = self.dim();
        let start = row * dim;
        &mut self.as_tensor_mut().data_mut()[start..start + dim]
    }

    #[inline]
    pub fn get_vec_owned(&self, i: isize) -> Vec<T>
    where
        T: Copy,
    {
        self.get_vec(i).to_vec()
    }

    #[inline]
    pub fn get_vector(&self, i: isize) -> &[T] {
        self.get_vec(i)
    }

    #[inline]
    pub fn get_vector_mut(&mut self, i: isize) -> &mut [T] {
        self.get_vec_mut(i)
    }

    #[inline]
    pub fn set_vec(&mut self, i: isize, values: &[T])
    where
        T: Copy,
    {
        assert_eq!(
            values.len(),
            self.dim(),
            "VectorList::set_vec: vector length mismatch"
        );
        self.get_vec_mut(i).copy_from_slice(values);
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
    pub fn get_axis(&self, k: isize) -> Vec<T>
    where
        T: Copy,
    {
        self.axis(k)
    }

    #[inline]
    pub fn set_axis_from_slice(&mut self, k: isize, values: &[T])
    where
        T: Copy,
    {
        assert_eq!(
            values.len(),
            self.num_vecs(),
            "VectorList::set_axis_from_slice: length mismatch"
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
    pub fn set_vector_from_slice(&mut self, i: isize, values: &[T])
    where
        T: Copy,
    {
        self.set_vec(i, values);
    }

    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.tensor.fill(value);
    }

    pub fn print(&self)
    where
        T: Copy,
    {
        for i in 0..self.num_vecs() {
            println!("{:?}", self.get_vec(i as isize));
        }
    }

    pub fn par_for_each_vec<F>(&self, f: F)
    where
        T: Sync,
        F: Fn(usize, &[T]) + Send + Sync,
    {
        self.as_tensor()
            .data()
            .par_chunks_exact(self.dim())
            .enumerate()
            .for_each(|(i, row)| f(i, row));
    }

    pub fn par_for_each_vec_mut<F>(&mut self, f: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Send + Sync,
    {
        let dim = self.dim();
        self.as_tensor_mut()
            .data_mut()
            .par_chunks_exact_mut(dim)
            .enumerate()
            .for_each(|(i, row)| f(i, row));
    }

    pub fn scale_vectors_by_list(&mut self, scales: &[T])
    where
        T: Copy + Send + Sync,
    {
        assert_eq!(
            scales.len(),
            self.num_vecs(),
            "VectorList::scale_vectors_by_list: length mismatch"
        );
        self.par_for_each_vec_mut(|i, row| {
            let scale = scales[i];
            row.iter_mut().for_each(|x| *x = *x * scale);
        });
    }

    pub fn get_norms(&self) -> Vec<T>
    where
        T: Copy + Send + Sync,
    {
        self.as_tensor()
            .data()
            .par_chunks_exact(self.dim())
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
        let norms = self.get_norms();
        let scales: Vec<T> = norms
            .par_iter()
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
        let norms = self.get_norms();
        let mut units = self.clone();
        let scales: Vec<T> = norms
            .par_iter()
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

    pub fn from_ndarray(array: &Array2<T>) -> Self
    where
        T: Copy,
    {
        let shape = array.shape();
        assert_eq!(shape.len(), 2, "VectorList::from_ndarray expects rank 2");
        assert!(
            shape[0] > 0 && shape[1] > 0,
            "VectorList::from_ndarray: shape must be nonzero"
        );
        Self::from_vec(shape[1], shape[0], array.iter().copied().collect())
    }

    pub fn to_ndarray(&self) -> Array2<T>
    where
        T: Copy,
    {
        Array2::from_shape_vec(
            (self.num_vecs(), self.dim()),
            self.as_tensor().data().to_vec(),
        )
        .expect("VectorList::to_ndarray: shape/data length mismatch")
    }
}

impl<T> VectorList<T>
where
    T: Scalar + Serialize + Copy,
{
    #[inline]
    pub fn serialize_value(&self) -> Result<Value, serde_json::Error> {
        self.to_json_value()
    }

    #[inline]
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.to_json_string()
    }
}

impl<T: Scalar + Copy> NdarrayConvert for VectorList<T> {
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
                    .copied()
                    .map(|x| x $op rhs)
                    .collect();
                VectorList::from_vec(self.dim(), self.num_vecs(), data)
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

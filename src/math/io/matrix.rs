//! IO and external-format interop for matrix containers.

use ndarray::Array2;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::math::io::json::{FlatPayload, FlatPayloadRef, FromJsonPayload, ToJsonPayload};
use crate::math::io::ndarray::NdarrayConvert;
use crate::math::scalar::Scalar;
use crate::math::tensor::rank_2::matrix::{Matrix, MatrixBackend, RankNDense, RankNSparse};

impl<T, B> Serialize for Matrix<T, B>
where
    T: Scalar + Serialize + Copy,
    B: MatrixBackend<T>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if let Some(data) = self.contiguous_data() {
            let shape = self.shape();
            FlatPayloadRef {
                kind: "matrix",
                shape: &shape,
                data,
            }
            .serialize(serializer)
        } else {
            self.to_json_payload()
                .map_err(serde::ser::Error::custom)?
                .serialize(serializer)
        }
    }
}

impl<'de, T> Deserialize<'de> for Matrix<T, RankNDense<T>>
where
    T: Scalar + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = FlatPayload::<T>::deserialize(deserializer)?;
        <Self as FromJsonPayload>::from_json_payload(payload).map_err(serde::de::Error::custom)
    }
}

impl<'de, T> Deserialize<'de> for Matrix<T, RankNSparse<T>>
where
    T: Scalar + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = FlatPayload::<T>::deserialize(deserializer)?;
        <Self as FromJsonPayload>::from_json_payload(payload).map_err(serde::de::Error::custom)
    }
}

impl<T, B> ToJsonPayload for Matrix<T, B>
where
    T: Scalar + Serialize + Copy,
    B: MatrixBackend<T>,
{
    type Payload = FlatPayload<T>;

    fn to_json_payload(&self) -> Result<Self::Payload, serde_json::Error> {
        let rows = self.rows();
        let cols = self.cols();
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                data.push(self.get(row as isize, col as isize));
            }
        }
        Ok(FlatPayload::new("matrix", vec![rows, cols], data))
    }
}

impl<T> FromJsonPayload for Matrix<T, RankNDense<T>>
where
    T: Scalar + DeserializeOwned,
{
    type Payload = FlatPayload<T>;

    fn from_json_payload(payload: Self::Payload) -> Result<Self, String> {
        let (rows, cols, data) = matrix_payload_parts(payload)?;
        Ok(Self::from_vec(rows, cols, data))
    }
}

impl<T> FromJsonPayload for Matrix<T, RankNSparse<T>>
where
    T: Scalar + DeserializeOwned,
{
    type Payload = FlatPayload<T>;

    fn from_json_payload(payload: Self::Payload) -> Result<Self, String> {
        let (rows, cols, data) = matrix_payload_parts(payload)?;
        let triplets = data
            .into_iter()
            .enumerate()
            .map(|(idx, value)| (idx / cols, idx % cols, value));
        Ok(Self::from_triplets(rows, cols, triplets))
    }
}

fn matrix_payload_parts<T>(payload: FlatPayload<T>) -> Result<(usize, usize, Vec<T>), String> {
    payload.validate_dense("matrix")?;
    if payload.shape.len() != 2 {
        return Err(format!(
            "matrix shape rank mismatch: expected 2, got {}",
            payload.shape.len()
        ));
    }
    Ok((payload.shape[0], payload.shape[1], payload.data))
}

impl<T, B> Matrix<T, B>
where
    T: Scalar + Serialize + Copy,
    B: MatrixBackend<T>,
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

impl<T, B> Matrix<T, B>
where
    T: Scalar + Copy,
    B: MatrixBackend<T>,
{
    pub fn to_ndarray(&self) -> Array2<T> {
        let rows = self.rows();
        let cols = self.cols();
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                data.push(self.get(row as isize, col as isize));
            }
        }
        Array2::from_shape_vec((rows, cols), data)
            .expect("Matrix::to_ndarray: shape/data length mismatch")
    }
}

impl<T: Scalar> Matrix<T, RankNDense<T>> {
    pub fn from_ndarray(array: &Array2<T>) -> Self {
        let shape = array.shape();
        assert_eq!(shape.len(), 2, "Matrix::from_ndarray expects rank 2");
        assert!(
            shape[0] > 0 && shape[1] > 0,
            "Matrix::from_ndarray: shape must be nonzero"
        );
        Self::from_vec(shape[0], shape[1], array.iter().copied().collect())
    }
}

impl<T: Scalar> Matrix<T, RankNSparse<T>> {
    pub fn from_ndarray(array: &Array2<T>) -> Self {
        let dense = Matrix::<T, RankNDense<T>>::from_ndarray(array);
        dense.to_sparse()
    }
}

impl<T: Scalar> NdarrayConvert for Matrix<T, RankNDense<T>> {
    type NdArray = Array2<T>;

    #[inline]
    fn from_ndarray(array: &Self::NdArray) -> Self {
        Matrix::<T, RankNDense<T>>::from_ndarray(array)
    }

    #[inline]
    fn to_ndarray(&self) -> Self::NdArray {
        Matrix::<T, RankNDense<T>>::to_ndarray(self)
    }
}

impl<T: Scalar> NdarrayConvert for Matrix<T, RankNSparse<T>> {
    type NdArray = Array2<T>;

    #[inline]
    fn from_ndarray(array: &Self::NdArray) -> Self {
        Matrix::<T, RankNSparse<T>>::from_ndarray(array)
    }

    #[inline]
    fn to_ndarray(&self) -> Self::NdArray {
        Matrix::<T, RankNSparse<T>>::to_ndarray(self)
    }
}

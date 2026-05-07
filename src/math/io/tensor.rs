//! JSON and serde integration for tensor storage and facade types.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::math::io::json::{FlatPayload, FromJsonPayload, ToJsonPayload};
use crate::math::scalar::Scalar;
use crate::math::tensor::rank_n::{
    Dense, Sparse, Tensor, dense::Tensor as DenseStorage, sparse::Tensor as SparseStorage,
    tensor_trait::TensorTrait,
};

impl<T> Serialize for DenseStorage<T>
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

impl<'de, T> Deserialize<'de> for DenseStorage<T>
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

impl<T> ToJsonPayload for DenseStorage<T>
where
    T: Scalar + Serialize + Copy,
{
    type Payload = FlatPayload<T>;

    fn to_json_payload(&self) -> Result<Self::Payload, serde_json::Error> {
        Ok(FlatPayload::new(
            "tensor",
            self.shape().to_vec(),
            self.data().to_vec(),
        ))
    }
}

impl<T> FromJsonPayload for DenseStorage<T>
where
    T: Scalar + DeserializeOwned,
{
    type Payload = FlatPayload<T>;

    fn from_json_payload(payload: Self::Payload) -> Result<Self, String> {
        payload.validate_dense("tensor")?;
        Ok(Self::from_parts_unchecked(payload.shape, payload.data))
    }
}

impl<T> Serialize for SparseStorage<T>
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

impl<'de, T> Deserialize<'de> for SparseStorage<T>
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

impl<T> ToJsonPayload for SparseStorage<T>
where
    T: Scalar + Serialize + Copy,
{
    type Payload = FlatPayload<T>;

    fn to_json_payload(&self) -> Result<Self::Payload, serde_json::Error> {
        let dense = self.to_dense();
        Ok(FlatPayload::new(
            "tensor_sparse",
            dense.shape().to_vec(),
            dense.data().to_vec(),
        ))
    }
}

impl<T> FromJsonPayload for SparseStorage<T>
where
    T: Scalar + DeserializeOwned,
{
    type Payload = FlatPayload<T>;

    fn from_json_payload(payload: Self::Payload) -> Result<Self, String> {
        payload.validate_dense("tensor_sparse")?;
        let dense = DenseStorage::from_parts_unchecked(payload.shape, payload.data);
        Ok(Self::from_dense(&dense))
    }
}

impl<T> Serialize for Tensor<T, Dense>
where
    T: Scalar + Serialize + Copy,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_json_value()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for Tensor<T, Dense>
where
    T: Scalar + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner = DenseStorage::<T>::deserialize(deserializer)?;
        Ok(Self::from_storage(inner))
    }
}

impl<T> Serialize for Tensor<T, Sparse>
where
    T: Scalar + Serialize + Copy,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_json_value()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for Tensor<T, Sparse>
where
    T: Scalar + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner = SparseStorage::<T>::deserialize(deserializer)?;
        Ok(Self::from_storage(inner))
    }
}

impl<T> Tensor<T, Dense>
where
    T: Scalar + Serialize + Copy,
{
    pub fn to_json_value(&self) -> Result<Value, serde_json::Error> {
        self.storage().to_json_value()
    }

    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        self.storage().to_json_string()
    }
}

impl<T> Tensor<T, Sparse>
where
    T: Scalar + Serialize + Copy,
{
    pub fn to_json_value(&self) -> Result<Value, serde_json::Error> {
        self.storage().to_json_value()
    }

    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        self.storage().to_json_string()
    }
}

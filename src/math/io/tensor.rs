//! JSON and serde integration for tensor storage and facade types.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::math::io::json::{
    FlatPayload, FlatPayloadRef, FromJsonPayload, SparsePayload, ToJsonPayload, ensure_finite,
};
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
        ensure_finite(self.data(), "tensor").map_err(serde::ser::Error::custom)?;
        FlatPayloadRef::new("tensor", self.shape(), self.data()).serialize(serializer)
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
        ensure_finite(self.data(), "tensor")
            .map_err(|error| serde_json::Error::io(std::io::Error::other(error)))?;
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
        ensure_finite(&payload.data, "tensor")?;
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
        sparse_payload(self, "tensor_sparse")
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
        let payload = SparsePayload::<T>::deserialize(deserializer)?;
        <Self as FromJsonPayload>::from_json_payload(payload).map_err(serde::de::Error::custom)
    }
}

impl<T> ToJsonPayload for SparseStorage<T>
where
    T: Scalar + Serialize + Copy,
{
    type Payload = SparsePayload<T>;

    fn to_json_payload(&self) -> Result<Self::Payload, serde_json::Error> {
        sparse_payload(self, "tensor_sparse")
            .map_err(|error| serde_json::Error::io(std::io::Error::other(error)))
    }
}

impl<T> FromJsonPayload for SparseStorage<T>
where
    T: Scalar + DeserializeOwned,
{
    type Payload = SparsePayload<T>;

    fn from_json_payload(payload: Self::Payload) -> Result<Self, String> {
        let (shape, indices, values) = payload.into_validated_parts("tensor_sparse")?;
        ensure_finite(&values, "tensor_sparse")?;
        if let Some(position) = values.iter().position(|value| *value == T::zero()) {
            return Err(format!(
                "tensor_sparse contains an explicit zero at sparse position {position}"
            ));
        }
        Ok(Self::from_flat_pairs(
            shape,
            indices.into_iter().zip(values).collect(),
        ))
    }
}

/// Builds one deterministic sparse payload without materializing implicit zeros.
fn sparse_payload<T>(storage: &SparseStorage<T>, kind: &str) -> Result<SparsePayload<T>, String>
where
    T: Scalar + Serialize + Copy,
{
    let mut entries = storage
        .iter()
        .map(|(&index, &value)| (index, value))
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(index, _)| *index);
    let mut indices = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    for (index, value) in entries {
        if !value.is_finite() {
            return Err(format!(
                "{kind} contains a non-finite scalar at flat index {index}"
            ));
        }
        indices.push(index);
        values.push(value);
    }
    Ok(SparsePayload::new(
        kind,
        storage.shape().to_vec(),
        indices,
        values,
    ))
}

impl<T> Serialize for Tensor<T, Dense>
where
    T: Scalar + Serialize + Copy,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.storage().serialize(serializer)
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
        self.storage().serialize(serializer)
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

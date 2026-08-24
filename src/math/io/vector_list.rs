//! IO and external-format interop for vector-list containers.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::math::io::json::{
    FlatPayload, FlatPayloadRef, FromJsonPayload, ToJsonPayload, ensure_finite,
};
use crate::math::scalar::Scalar;
use crate::math::tensor::rank_2::vector_list::VectorList;

impl<T> Serialize for VectorList<T>
where
    T: Scalar + Serialize + Copy,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let shape = [self.num_vectors(), self.dim()];
        ensure_finite(self.as_tensor().data(), "vector_list").map_err(serde::ser::Error::custom)?;
        FlatPayloadRef::new("vector_list", &shape, self.as_tensor().data()).serialize(serializer)
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
        ensure_finite(self.as_tensor().data(), "vector_list")
            .map_err(|error| serde_json::Error::io(std::io::Error::other(error)))?;
        Ok(FlatPayload::new(
            "vector_list",
            vec![self.num_vectors(), self.dim()],
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
        ensure_finite(&payload.data, "vector_list")?;
        if payload.shape.len() != 2 {
            return Err(format!(
                "vector_list shape rank mismatch: expected 2, got {}",
                payload.shape.len()
            ));
        }
        Ok(Self::from_vec(
            payload.shape[1],
            payload.shape[0],
            payload.data,
        ))
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

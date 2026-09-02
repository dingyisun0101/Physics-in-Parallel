//! IO and external-format interop for vector-list containers.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::math::io::json::{FlatPayload, FlatPayloadRef, ensure_finite};
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
        vector_list_from_payload(payload).map_err(serde::de::Error::custom)
    }
}

fn vector_list_from_payload<T>(payload: FlatPayload<T>) -> Result<VectorList<T>, String>
where
    T: Scalar + DeserializeOwned + Copy,
{
    payload.validate_dense("vector_list")?;
    ensure_finite(&payload.data, "vector_list")?;
    if payload.shape.len() != 2 {
        return Err(format!(
            "vector_list shape rank mismatch: expected 2, got {}",
            payload.shape.len()
        ));
    }
    Ok(VectorList::from_vec(
        payload.shape[1],
        payload.shape[0],
        payload.data,
    ))
}

//! Direct, validated Serde for square lattices.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::math::Scalar;
use crate::rng::ResolvedRng;
use crate::space::{SquareLattice, SquareLatticeGeometry};

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct SquareLatticeRef<'a, T> {
    geometry: &'a SquareLatticeGeometry,
    values: &'a [T],
    initialization_rng: Option<ResolvedRng>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SquareLatticeDocument<T> {
    geometry: SquareLatticeGeometry,
    values: Vec<T>,
    initialization_rng: Option<ResolvedRng>,
}

impl<T> Serialize for SquareLattice<T>
where
    T: Scalar + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SquareLatticeRef {
            geometry: self.geometry(),
            values: self.data(),
            initialization_rng: self.initialization_resolved_rng(),
        }
        .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for SquareLattice<T>
where
    T: Scalar + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let document = SquareLatticeDocument::<T>::deserialize(deserializer)?;
        if document.values.len() != document.geometry.num_sites() {
            return Err(serde::de::Error::custom(format!(
                "square lattice requires {} values, got {}",
                document.geometry.num_sites(),
                document.values.len()
            )));
        }
        Ok(SquareLattice::from_parts(
            document.geometry,
            document.values,
            document.initialization_rng,
        ))
    }
}

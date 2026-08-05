//! IO and external-format interop for square lattices.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use ndarray::{ArrayD, IxDyn};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::math::io::json::{FlatPayload, FlatPayloadRef, FromJsonPayload, ToJsonPayload};
use crate::math::prelude::{Scalar, ScalarSerde};
use crate::space::discrete::square_lattice::{
    BoundaryCondition, SquareLattice, SquareLatticeConfig, VacancyValue,
};

impl BoundaryCondition {
    #[inline]
    pub(crate) fn kind_tag(self) -> &'static str {
        match self {
            Self::Periodic => "square_lattice_periodic",
            Self::Reflective => "square_lattice_reflective",
        }
    }

    #[inline]
    pub(crate) fn from_kind_tag(kind: &str) -> Result<Self, String> {
        match kind {
            "square_lattice_periodic" => Ok(Self::Periodic),
            "square_lattice_reflective" => Ok(Self::Reflective),
            _ => Err(format!(
                "square lattice kind must be 'square_lattice_periodic' or 'square_lattice_reflective'; got '{kind}'"
            )),
        }
    }
}

impl<T> Serialize for SquareLattice<T>
where
    T: Scalar + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        FlatPayloadRef {
            kind: self.cfg.boundary.kind_tag(),
            shape: self.cfg.shape(),
            data: self.data(),
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
        let payload = FlatPayload::<T>::deserialize(deserializer)?;
        <Self as FromJsonPayload>::from_json_payload(payload).map_err(serde::de::Error::custom)
    }
}

impl<T> ToJsonPayload for SquareLattice<T>
where
    T: Scalar + Serialize,
{
    type Payload = FlatPayload<T>;

    fn to_json_payload(&self) -> Result<Self::Payload, serde_json::Error> {
        Ok(FlatPayload::new(
            self.cfg.boundary.kind_tag(),
            self.tensor_shape(),
            self.data().to_vec(),
        ))
    }
}

impl<T> FromJsonPayload for SquareLattice<T>
where
    T: Scalar + DeserializeOwned,
{
    type Payload = FlatPayload<T>;

    fn from_json_payload(payload: Self::Payload) -> Result<Self, String> {
        let boundary = BoundaryCondition::from_kind_tag(&payload.kind)?;
        let expected_len = payload.validate_shape("lattice")?;
        if payload.data.len() != expected_len {
            return Err(format!(
                "lattice data length mismatch: expected {expected_len}, got {}",
                payload.data.len()
            ));
        }

        let cfg = SquareLatticeConfig::new(&payload.shape, boundary);
        Ok(SquareLattice::from_parts(cfg, payload.data))
    }
}

impl<T: Scalar + Clone> SquareLattice<T> {
    pub fn from_ndarray(array: &ArrayD<T>, boundary: BoundaryCondition) -> Self {
        let owned = array.to_owned();
        let shape = owned.shape().to_vec();
        assert!(
            !shape.is_empty(),
            "SquareLattice::from_ndarray: shape must be non-empty"
        );
        let (data, _) = owned.into_raw_vec_and_offset();
        SquareLattice::from_parts(SquareLatticeConfig::new(&shape, boundary), data)
    }

    pub fn to_ndarray(&self) -> ArrayD<T> {
        ArrayD::from_shape_vec(IxDyn(&self.tensor_shape()), self.data().to_vec())
            .expect("SquareLattice::to_ndarray: shape/data length mismatch")
    }
}

impl<T: ScalarSerde + VacancyValue> SquareLattice<T> {
    #[inline]
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.to_json_string()
    }
}

pub fn save_square_lattice<T>(
    lattice: &SquareLattice<T>,
    target_shape: &[usize],
    output_file: &PathBuf,
) -> std::io::Result<()>
where
    T: ScalarSerde + VacancyValue,
{
    let lattice_to_save = lattice.downsample(target_shape);
    let shape = lattice_to_save.tensor_shape();
    let json_data = FlatPayloadRef {
        kind: lattice_to_save.cfg.boundary.kind_tag(),
        shape: &shape,
        data: lattice_to_save.data(),
    };
    let json = serde_json::to_string_pretty(&json_data).expect("failed to serialize lattice");

    let mut file = File::create(output_file)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

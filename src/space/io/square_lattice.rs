//! IO and external-format interop for square lattices.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::math::io::json::{
    FlatPayload, FlatPayloadRef, FromJsonPayload, ToJsonPayload, ensure_finite,
};
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
        ensure_finite(self.data(), "square_lattice").map_err(serde::ser::Error::custom)?;
        FlatPayloadRef::new(self.cfg.boundary.kind_tag(), self.cfg.shape(), self.data())
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
        ensure_finite(self.data(), "square_lattice")
            .map_err(|error| serde_json::Error::io(std::io::Error::other(error)))?;
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
        payload.validate_version("square_lattice")?;
        payload.validate_scalar("square_lattice")?;
        let boundary = BoundaryCondition::from_kind_tag(&payload.kind)?;
        let expected_len = payload.validate_shape("lattice")?;
        if payload.data.len() != expected_len {
            return Err(format!(
                "lattice data length mismatch: expected {expected_len}, got {}",
                payload.data.len()
            ));
        }
        ensure_finite(&payload.data, "square_lattice")?;

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

pub fn save_square_lattice<T, P>(
    lattice: &SquareLattice<T>,
    target_shape: &[usize],
    output_file: P,
) -> std::io::Result<()>
where
    T: ScalarSerde + VacancyValue,
    P: AsRef<Path>,
{
    let lattice_to_save = lattice.downsample(target_shape);
    let shape = lattice_to_save.tensor_shape();
    ensure_finite(lattice_to_save.data(), "square_lattice").map_err(std::io::Error::other)?;
    let json_data = FlatPayloadRef::new(
        lattice_to_save.cfg.boundary.kind_tag(),
        &shape,
        lattice_to_save.data(),
    );
    let mut writer = BufWriter::new(File::create(output_file)?);
    serde_json::to_writer(&mut writer, &json_data).map_err(std::io::Error::other)?;
    writer.flush()
}

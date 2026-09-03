use std::any::TypeId;

use num_complex::Complex;
use serde::{Deserialize, Serialize};

use crate::math::scalar::Scalar;

/// Version shared by the current PiP JSON payload schemas.
pub const JSON_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatPayload<T> {
    pub kind: String,
    pub version: u32,
    pub scalar: String,
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T: Scalar> FlatPayload<T> {
    pub(crate) fn new(kind: &str, shape: Vec<usize>, data: Vec<T>) -> Self {
        Self {
            kind: kind.to_string(),
            version: JSON_SCHEMA_VERSION,
            scalar: scalar_kind::<T>().to_owned(),
            shape,
            data,
        }
    }

    pub fn validate_version(&self, context: &str) -> Result<(), String> {
        if self.version != JSON_SCHEMA_VERSION {
            return Err(format!(
                "{context} version mismatch: expected {JSON_SCHEMA_VERSION}, got {}",
                self.version
            ));
        }
        Ok(())
    }

    pub fn validate_kind(&self, expected_kind: &str) -> Result<(), String> {
        if self.kind != expected_kind {
            return Err(format!(
                "{expected_kind} kind must be '{expected_kind}', got '{}'",
                self.kind
            ));
        }
        Ok(())
    }

    pub fn validate_scalar(&self, context: &str) -> Result<(), String> {
        let expected = scalar_kind::<T>();
        if self.scalar != expected {
            return Err(format!(
                "{context} scalar mismatch: expected '{expected}', got '{}'",
                self.scalar
            ));
        }
        Ok(())
    }
    pub fn validate_shape(&self, context: &str) -> Result<usize, String> {
        checked_num_elements(&self.shape, context)
    }

    pub fn validate_dense(&self, expected_kind: &str) -> Result<(), String> {
        self.validate_version(expected_kind)?;
        self.validate_kind(expected_kind)?;
        self.validate_scalar(expected_kind)?;
        let expected_len = self.validate_shape(expected_kind)?;
        if self.data.len() != expected_len {
            return Err(format!(
                "{expected_kind} data length mismatch: expected {expected_len}, got {}",
                self.data.len()
            ));
        }
        Ok(())
    }
}

/// Owned versioned representation of sparse row-major storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparsePayload<T> {
    pub kind: String,
    pub version: u32,
    pub scalar: String,
    pub shape: Vec<usize>,
    pub indices: Vec<usize>,
    pub values: Vec<T>,
}

/// Validated owned sparse components in `(shape, indices, values)` order.
pub type SparsePayloadParts<T> = (Vec<usize>, Vec<usize>, Vec<T>);

impl<T: Scalar> SparsePayload<T> {
    pub fn new(kind: &str, shape: Vec<usize>, indices: Vec<usize>, values: Vec<T>) -> Self {
        Self {
            kind: kind.to_owned(),
            version: JSON_SCHEMA_VERSION,
            scalar: scalar_kind::<T>().to_owned(),
            shape,
            indices,
            values,
        }
    }

    pub fn into_validated_parts(
        self,
        expected_kind: &str,
    ) -> Result<SparsePayloadParts<T>, String> {
        if self.version != JSON_SCHEMA_VERSION {
            return Err(format!(
                "{expected_kind} version mismatch: expected {JSON_SCHEMA_VERSION}, got {}",
                self.version
            ));
        }
        if self.kind != expected_kind {
            return Err(format!(
                "{expected_kind} kind must be '{expected_kind}', got '{}'",
                self.kind
            ));
        }
        let expected_scalar = scalar_kind::<T>();
        if self.scalar != expected_scalar {
            return Err(format!(
                "{expected_kind} scalar mismatch: expected '{expected_scalar}', got '{}'",
                self.scalar
            ));
        }
        let logical_size = checked_num_elements(&self.shape, expected_kind)?;
        if self.indices.len() != self.values.len() {
            return Err(format!(
                "{expected_kind} sparse entry length mismatch: {} indices, {} values",
                self.indices.len(),
                self.values.len()
            ));
        }
        let mut previous = None;
        for &index in &self.indices {
            if index >= logical_size {
                return Err(format!(
                    "{expected_kind} sparse index out of bounds: {index} >= {logical_size}"
                ));
            }
            if previous.is_some_and(|prior| prior >= index) {
                return Err(format!(
                    "{expected_kind} sparse indices must be strictly increasing"
                ));
            }
            previous = Some(index);
        }
        Ok((self.shape, self.indices, self.values))
    }
}

pub fn checked_num_elements(shape: &[usize], context: &str) -> Result<usize, String> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(format!(
            "{context} shape must contain only nonzero dimensions"
        ));
    }

    let mut expected_len = 1usize;
    for &dim in shape {
        expected_len = expected_len
            .checked_mul(dim)
            .ok_or_else(|| format!("{context} shape product overflow: {shape:?}"))?;
    }
    Ok(expected_len)
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct FlatPayloadRef<'a, T> {
    pub(crate) kind: &'a str,
    pub(crate) version: u32,
    pub(crate) scalar: &'static str,
    pub(crate) shape: &'a [usize],
    pub(crate) data: &'a [T],
}

impl<'a, T: Scalar> FlatPayloadRef<'a, T> {
    pub(crate) fn new(kind: &'a str, shape: &'a [usize], data: &'a [T]) -> Self {
        Self {
            kind,
            version: JSON_SCHEMA_VERSION,
            scalar: scalar_kind::<T>(),
            shape,
            data,
        }
    }
}

/// Stable JSON identifier for every scalar admitted by PiP's sealed contract.
pub fn scalar_kind<T: Scalar>() -> &'static str {
    let type_id = TypeId::of::<T>();
    macro_rules! identify {
        ($($scalar:ty => $kind:literal),+ $(,)?) => {
            $(if type_id == TypeId::of::<$scalar>() {
                return $kind;
            })+
        };
    }
    identify!(
        f32 => "f32",
        f64 => "f64",
        i8 => "i8",
        i16 => "i16",
        i32 => "i32",
        i64 => "i64",
        i128 => "i128",
        isize => "isize",
        u8 => "u8",
        u16 => "u16",
        u32 => "u32",
        u64 => "u64",
        u128 => "u128",
        usize => "usize",
        Complex<f32> => "complex_f32",
        Complex<f64> => "complex_f64",
    );
    unreachable!("Scalar is sealed to types covered by scalar_kind")
}

/// Rejects values that ordinary JSON numbers cannot represent faithfully.
pub(crate) fn ensure_finite<T: Scalar>(values: &[T], context: &str) -> Result<(), String> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(format!(
            "{context} contains a non-finite scalar at flat index {index}"
        ));
    }
    Ok(())
}

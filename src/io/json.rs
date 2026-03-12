use serde::{Deserialize, Serialize};
use serde_json::Value;

pub trait ToJsonPayload {
    type Payload: Serialize;

    fn to_json_payload(&self) -> Result<Self::Payload, serde_json::Error>;

    fn to_json_value(&self) -> Result<Value, serde_json::Error> {
        serde_json::to_value(self.to_json_payload()?)
    }

    fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.to_json_payload()?)
    }
}

pub trait FromJsonPayload: Sized {
    type Payload: for<'de> Deserialize<'de>;

    fn from_json_payload(payload: Self::Payload) -> Result<Self, String>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatPayload<T> {
    pub kind: String,
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> FlatPayload<T> {
    pub fn new(kind: &str, shape: Vec<usize>, data: Vec<T>) -> Self {
        Self {
            kind: kind.to_string(),
            shape,
            data,
        }
    }

    pub fn validate_kind(&self, expected_kind: &str) -> Result<(), String> {
        if self.kind != expected_kind {
            return Err(format!("{expected_kind} kind must be '{expected_kind}'"));
        }
        Ok(())
    }
    pub fn validate_shape(&self, context: &str) -> Result<usize, String> {
        checked_num_elements(&self.shape, context)
    }

    pub fn validate_dense(&self, expected_kind: &str) -> Result<(), String> {
        self.validate_kind(expected_kind)?;
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

pub fn checked_num_elements(shape: &[usize], context: &str) -> Result<usize, String> {
    if shape.is_empty() || shape.iter().any(|&dim| dim == 0) {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledPayload {
    pub label: String,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttrsCorePayload {
    pub n_objects: Option<usize>,
    pub num_attrs: usize,
    pub attrs: Vec<LabeledPayload>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysObjPayload {
    pub meta: Value,
    pub core: AttrsCorePayload,
}

#[derive(Debug, Clone, Serialize)]
pub struct FlatPayloadRef<'a, T> {
    pub kind: &'a str,
    pub shape: &'a [usize],
    pub data: &'a [T],
}

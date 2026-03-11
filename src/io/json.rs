use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::math::scalar::Scalar;

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
pub struct DenseTensorPayload<T> {
    pub kind: String,
    pub scalar_type: String,
    pub shape: Vec<usize>,
    pub storage: String,
    pub data: Vec<T>,
}

impl<T> DenseTensorPayload<T> {
    pub fn new(kind: &str, scalar_type: String, shape: Vec<usize>, data: Vec<T>) -> Self {
        Self {
            kind: kind.to_string(),
            scalar_type,
            shape,
            storage: "dense".to_string(),
            data,
        }
    }

    pub fn validate<TScalar: Scalar>(&self, expected_kind: &str) -> Result<(), String> {
        if self.kind != expected_kind {
            return Err(format!("{expected_kind} kind must be '{expected_kind}'"));
        }
        if self.storage != "dense" {
            return Err(format!("{expected_kind} storage must be 'dense'"));
        }
        if self.scalar_type != std::any::type_name::<TScalar>() {
            return Err(format!(
                "{expected_kind} scalar_type mismatch: expected {}, got {}",
                std::any::type_name::<TScalar>(),
                self.scalar_type
            ));
        }
        if self.shape.is_empty() || self.shape.iter().any(|&dim| dim == 0) {
            return Err(format!(
                "{expected_kind} shape must contain only nonzero dimensions"
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseEntry<T> {
    pub index: usize,
    pub value: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseData<T> {
    pub nnz: usize,
    pub entries: Vec<SparseEntry<T>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseTensorPayload<T> {
    pub kind: String,
    pub scalar_type: String,
    pub shape: Vec<usize>,
    pub storage: String,
    pub data: SparseData<T>,
}

impl<T> SparseTensorPayload<T> {
    pub fn new(scalar_type: String, shape: Vec<usize>, entries: Vec<SparseEntry<T>>) -> Self {
        let nnz = entries.len();
        Self {
            kind: "tensor".to_string(),
            scalar_type,
            shape,
            storage: "sparse".to_string(),
            data: SparseData { nnz, entries },
        }
    }

    pub fn validate<TScalar: Scalar>(&self) -> Result<(), String> {
        if self.kind != "tensor" {
            return Err("sparse tensor kind must be 'tensor'".to_string());
        }
        if self.storage != "sparse" {
            return Err("sparse tensor storage must be 'sparse'".to_string());
        }
        if self.scalar_type != std::any::type_name::<TScalar>() {
            return Err(format!(
                "sparse tensor scalar_type mismatch: expected {}, got {}",
                std::any::type_name::<TScalar>(),
                self.scalar_type
            ));
        }
        if self.shape.is_empty() || self.shape.iter().any(|&dim| dim == 0) {
            return Err("sparse tensor shape must contain only nonzero dimensions".to_string());
        }
        if self.data.nnz != self.data.entries.len() {
            return Err("sparse tensor nnz must match entry count".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rank2RowsPayload<T> {
    pub kind: String,
    pub scalar_type: String,
    pub shape: [usize; 2],
    pub storage: String,
    pub data: Vec<Vec<T>>,
}

impl<T> Rank2RowsPayload<T> {
    pub fn new(kind: &str, scalar_type: String, shape: [usize; 2], data: Vec<Vec<T>>) -> Self {
        Self {
            kind: kind.to_string(),
            scalar_type,
            shape,
            storage: "dense".to_string(),
            data,
        }
    }

    pub fn validate<TScalar: Scalar>(&self, expected_kind: &str) -> Result<(), String> {
        if self.kind != expected_kind {
            return Err(format!("{expected_kind} kind must be '{expected_kind}'"));
        }
        if self.storage != "dense" {
            return Err(format!("{expected_kind} storage must be 'dense'"));
        }
        if self.scalar_type != std::any::type_name::<TScalar>() {
            return Err(format!(
                "{expected_kind} scalar_type mismatch: expected {}, got {}",
                std::any::type_name::<TScalar>(),
                self.scalar_type
            ));
        }
        let [rows, cols] = self.shape;
        if rows == 0 || cols == 0 {
            return Err(format!(
                "{expected_kind} shape must contain only nonzero dimensions"
            ));
        }
        if self.data.len() != rows {
            return Err(format!(
                "{expected_kind} row count mismatch: expected {rows}, got {}",
                self.data.len()
            ));
        }
        for row in &self.data {
            if row.len() != cols {
                return Err(format!(
                    "{expected_kind} column count mismatch: expected {cols}, got {}",
                    row.len()
                ));
            }
        }
        Ok(())
    }

    pub fn validate_vector_list<TScalar: Scalar>(&self) -> Result<(), String> {
        if self.kind != "vector_list" {
            return Err("vector_list kind must be 'vector_list'".to_string());
        }
        if self.storage != "dense" {
            return Err("vector_list storage must be 'dense'".to_string());
        }
        if self.scalar_type != std::any::type_name::<TScalar>() {
            return Err(format!(
                "vector_list scalar_type mismatch: expected {}, got {}",
                std::any::type_name::<TScalar>(),
                self.scalar_type
            ));
        }
        let [dim, n] = self.shape;
        if dim == 0 || n == 0 {
            return Err("vector_list shape must contain only nonzero dimensions".to_string());
        }
        if self.data.len() != n {
            return Err(format!(
                "vector_list row count mismatch: expected {n}, got {}",
                self.data.len()
            ));
        }
        for row in &self.data {
            if row.len() != dim {
                return Err(format!(
                    "vector_list dimension mismatch: expected {dim}, got {}",
                    row.len()
                ));
            }
        }
        Ok(())
    }
}

pub fn scalar_type_name<T: Scalar>() -> String {
    std::any::type_name::<T>().to_string()
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPayload<T> {
    pub kind: String,
    pub scalar_type: String,
    pub shape: [usize; 2],
    pub storage: String,
    pub data: Vec<T>,
}

impl<T> GridPayload<T> {
    pub fn new(scalar_type: String, shape: [usize; 2], data: Vec<T>) -> Self {
        Self {
            kind: "grid".to_string(),
            scalar_type,
            shape,
            storage: "dense".to_string(),
            data,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CompactGridPayload<'a, T> {
    pub shape: [usize; 2],
    pub data: &'a [T],
}

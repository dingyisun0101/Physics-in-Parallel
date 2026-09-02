//! IO and external-format interop for matrix containers.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::math::io::json::{FlatPayload, FlatPayloadRef, SparsePayload, ensure_finite};
use crate::math::scalar::Scalar;
use crate::math::tensor::rank_2::matrix::{Matrix, MatrixBackend, RankNDense, RankNSparse};

impl<T, B> Serialize for Matrix<T, B>
where
    T: Scalar + Serialize + Copy,
    B: MatrixBackend<T>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if let Some(data) = self.contiguous_data() {
            let shape = self.shape();
            ensure_finite(data, "matrix").map_err(serde::ser::Error::custom)?;
            FlatPayloadRef::new("matrix", &shape, data).serialize(serializer)
        } else if let Some(entries) = self.backend().sparse_entries() {
            sparse_matrix_payload(self.shape(), entries)
                .map_err(serde::ser::Error::custom)?
                .serialize(serializer)
        } else {
            let payload = logical_dense_matrix_payload(self).map_err(serde::ser::Error::custom)?;
            payload.serialize(serializer)
        }
    }
}

impl<'de, T> Deserialize<'de> for Matrix<T, RankNDense<T>>
where
    T: Scalar + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = FlatPayload::<T>::deserialize(deserializer)?;
        dense_matrix_from_payload(payload).map_err(serde::de::Error::custom)
    }
}

impl<'de, T> Deserialize<'de> for Matrix<T, RankNSparse<T>>
where
    T: Scalar + DeserializeOwned,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = SparsePayload::<T>::deserialize(deserializer)?;
        sparse_matrix_from_payload(payload).map_err(serde::de::Error::custom)
    }
}

fn dense_matrix_from_payload<T>(payload: FlatPayload<T>) -> Result<Matrix<T, RankNDense<T>>, String>
where
    T: Scalar + DeserializeOwned,
{
    let (rows, cols, data) = matrix_payload_parts(payload)?;
    Ok(Matrix::from_vec(rows, cols, data))
}

fn sparse_matrix_from_payload<T>(
    payload: SparsePayload<T>,
) -> Result<Matrix<T, RankNSparse<T>>, String>
where
    T: Scalar + DeserializeOwned,
{
    let (shape, indices, values) = payload.into_validated_parts("matrix_sparse")?;
    if shape.len() != 2 {
        return Err(format!(
            "matrix_sparse shape rank mismatch: expected 2, got {}",
            shape.len()
        ));
    }
    ensure_finite(&values, "matrix_sparse")?;
    if let Some(position) = values.iter().position(|value| *value == T::zero()) {
        return Err(format!(
            "matrix_sparse contains an explicit zero at sparse position {position}"
        ));
    }
    let rows = shape[0];
    let cols = shape[1];
    let triplets = indices
        .into_iter()
        .zip(values)
        .map(|(index, value)| (index / cols, index % cols, value));
    Ok(Matrix::from_triplets(rows, cols, triplets))
}

fn matrix_payload_parts<T: Scalar>(
    payload: FlatPayload<T>,
) -> Result<(usize, usize, Vec<T>), String> {
    payload.validate_dense("matrix")?;
    if payload.shape.len() != 2 {
        return Err(format!(
            "matrix shape rank mismatch: expected 2, got {}",
            payload.shape.len()
        ));
    }
    ensure_finite(&payload.data, "matrix")?;
    Ok((payload.shape[0], payload.shape[1], payload.data))
}

fn logical_dense_matrix_payload<T, B>(matrix: &Matrix<T, B>) -> Result<FlatPayload<T>, String>
where
    T: Scalar + Serialize + Copy,
    B: MatrixBackend<T>,
{
    let rows = matrix.rows();
    let cols = matrix.cols();
    let mut data = Vec::with_capacity(matrix.size());
    for row in 0..rows {
        for col in 0..cols {
            data.push(matrix.get(row as isize, col as isize));
        }
    }
    ensure_finite(&data, "matrix")?;
    Ok(FlatPayload::new("matrix", vec![rows, cols], data))
}

fn sparse_matrix_payload<T>(
    shape: [usize; 2],
    mut entries: Vec<(usize, T)>,
) -> Result<SparsePayload<T>, String>
where
    T: Scalar + Serialize + Copy,
{
    entries.sort_unstable_by_key(|(index, _)| *index);
    let mut indices = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    for (index, value) in entries {
        if !value.is_finite() {
            return Err(format!(
                "matrix_sparse contains a non-finite scalar at flat index {index}"
            ));
        }
        indices.push(index);
        values.push(value);
    }
    Ok(SparsePayload::new(
        "matrix_sparse",
        shape.to_vec(),
        indices,
        values,
    ))
}

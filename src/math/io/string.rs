//! String conversion helpers for math types.

use std::fmt::Display;
use std::str::FromStr;

use crate::math::scalar::Scalar;
use crate::math::tensor::rank_n::{
    Dense, Tensor, dense::Tensor as DenseStorage, tensor_trait::TensorTrait,
};

pub trait TensorStringConvert: Sized {
    fn to_tensor_string(&self) -> String;
    fn from_tensor_string(input: &str) -> Self;
}

pub(crate) fn format_dense_storage<T>(tensor: &DenseStorage<T>) -> String
where
    T: Scalar + Display,
{
    format_dense_parts(tensor.data(), tensor.shape())
}

fn format_dense_parts<T: Display>(data: &[T], shape: &[usize]) -> String {
    fn format_recursive<T: Display>(data: &[T], shape: &[usize]) -> String {
        if shape.len() == 1 {
            let mut line = String::new();
            for (i, value) in data.iter().take(shape[0]).enumerate() {
                line.push_str(&format!("{value}"));
                if i + 1 != shape[0] {
                    line.push(';');
                }
            }
            line
        } else {
            let stride = shape[1..].iter().product::<usize>();
            let mut result = String::new();
            for i in 0..shape[0] {
                let start = i * stride;
                let end = start + stride;
                let nested = format_recursive(&data[start..end], &shape[1..]);
                result.push('[');
                result.push_str(&nested);
                result.push(']');
                if i + 1 != shape[0] {
                    result.push(';');
                }
            }
            result
        }
    }

    format!("[{}]", format_recursive(data, shape))
}

fn parse_dense_2d<T>(input: &str) -> DenseStorage<T>
where
    T: Scalar + FromStr,
{
    let normalized = input
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .replace("];", "]|");

    let rows: Vec<Vec<T>> = normalized
        .split('|')
        .map(|row| {
            row.replace(['[', ']'], "")
                .split(';')
                .filter(|x| !x.trim().is_empty())
                .map(|num| {
                    let cleaned = num.trim();
                    cleaned
                        .parse::<T>()
                        .unwrap_or_else(|_| panic!("Invalid number in tensor: '{}'", cleaned))
                })
                .collect()
        })
        .collect();

    let num_rows = rows.len();
    assert!(num_rows > 0, "Tensor cannot be empty");

    let num_cols = rows[0].len();
    assert!(
        rows.iter().all(|row| row.len() == num_cols),
        "All rows must have the same number of columns"
    );

    DenseStorage::from_vec(&[num_rows, num_cols], rows.into_iter().flatten().collect())
}

impl<T> TensorStringConvert for DenseStorage<T>
where
    T: Scalar + Display + FromStr,
{
    fn to_tensor_string(&self) -> String {
        format_dense_storage(self)
    }

    fn from_tensor_string(input: &str) -> Self {
        parse_dense_2d(input)
    }
}

impl<T> TensorStringConvert for Tensor<T, Dense>
where
    T: Scalar + Display + FromStr,
{
    fn to_tensor_string(&self) -> String {
        format_dense_storage(self.storage())
    }

    fn from_tensor_string(input: &str) -> Self {
        Tensor::<T, Dense>::from_storage(parse_dense_2d(input))
    }
}

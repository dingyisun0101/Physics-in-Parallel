//! Shared tensor algorithms used by `TensorTrait` default methods.
//!
//! Backend implementations may override trait defaults when they can do better,
//! but these helpers define the common mathematical behavior.

use crate::math::scalar::Scalar;
use num_traits::Zero;
use rayon::prelude::*;

use super::tensor_trait::TensorTrait;

/// Dense logical size implied by a shape.
#[inline]
pub fn size(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Panic if two tensor shapes differ.
#[inline]
pub fn assert_same_shape<T, Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs)
where
    T: Scalar,
    Lhs: TensorTrait<T>,
    Rhs: TensorTrait<T>,
{
    assert_eq!(lhs.shape(), rhs.shape(), "Tensor shape mismatch");
}

/// Build a tensor by mapping every logical element.
#[inline]
pub fn map<T, A, F>(tensor: &A, f: F) -> A
where
    T: Scalar + Copy + Send + Sync,
    A: TensorTrait<T>,
    F: Fn(T) -> T + Sync + Send,
{
    let mut out = tensor.clone();
    out.par_map_in_place(f);
    out
}

/// Build a tensor by zipping two tensors with identical shape.
#[inline]
pub fn zip_with<T, Lhs, Rhs, F>(lhs: &Lhs, rhs: &Rhs, f: F) -> Lhs
where
    T: Scalar + Copy + Send + Sync,
    Lhs: TensorTrait<T>,
    Rhs: TensorTrait<T>,
    F: Fn(T, T) -> T + Sync + Send,
{
    assert_same_shape::<T, _, _>(lhs, rhs);
    let mut out = lhs.clone();
    out.par_zip_with_inplace(rhs, f);
    out
}

/// Type-preserving elementwise conjugate.
#[inline]
pub fn conj<T, A>(tensor: &A) -> A
where
    T: Scalar + Copy + Send + Sync,
    A: TensorTrait<T>,
{
    map(tensor, Scalar::conj)
}

/// Type-preserving elementwise absolute value.
#[inline]
pub fn abs<T, A>(tensor: &A) -> A
where
    T: Scalar + Copy + Send + Sync,
    A: TensorTrait<T>,
{
    map(tensor, Scalar::abs)
}

/// Type-preserving elementwise squared norm.
#[inline]
pub fn norm_sqr<T, A>(tensor: &A) -> A
where
    T: Scalar + Copy + Send + Sync,
    A: TensorTrait<T>,
{
    map(tensor, Scalar::norm_sqr)
}

/// Type-preserving elementwise square root.
#[inline]
pub fn sqrt<T, A>(tensor: &A) -> A
where
    T: Scalar + Copy + Send + Sync,
    A: TensorTrait<T>,
{
    map(tensor, Scalar::sqrt)
}

/// Type-preserving scalar multiplication.
#[inline]
pub fn scalar_mul<T, A>(tensor: &A, scalar: T) -> A
where
    T: Scalar + Copy + Send + Sync,
    A: TensorTrait<T>,
{
    map(tensor, move |x| x * scalar)
}

/// Type-preserving elementwise tensor multiplication.
#[inline]
pub fn elem_mul<T, Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> Lhs
where
    T: Scalar + Copy + Send + Sync,
    Lhs: TensorTrait<T>,
    Rhs: TensorTrait<T>,
{
    zip_with(lhs, rhs, |a, b| a * b)
}

/// Type-preserving elementwise tensor division.
#[inline]
pub fn elem_div<T, Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> Lhs
where
    T: Scalar + Copy + Send + Sync,
    Lhs: TensorTrait<T>,
    Rhs: TensorTrait<T>,
{
    zip_with(lhs, rhs, |a, b| a / b)
}

/// Type-preserving rank-2 transpose.
pub fn transpose<T, A>(tensor: &A) -> A
where
    T: Scalar + Copy + Send + Sync,
    A: TensorTrait<T>,
{
    assert_eq!(tensor.rank(), 2, "transpose requires a rank-2 tensor");
    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    let entries: Vec<(Vec<isize>, T)> = (0..rows * cols)
        .into_par_iter()
        .map(|k| {
            let i = k / cols;
            let j = k % cols;
            (
                vec![j as isize, i as isize],
                tensor.get(&[i as isize, j as isize]),
            )
        })
        .collect();

    let mut out = A::empty(&[cols, rows]);
    for (idx, value) in entries {
        out.set(&idx, value);
    }
    out
}

/// Type-preserving rank-2 Hermitian transpose.
pub fn hermitian_transpose<T, A>(tensor: &A) -> A
where
    T: Scalar + Copy + Send + Sync,
    A: TensorTrait<T>,
{
    assert_eq!(
        tensor.rank(),
        2,
        "hermitian_transpose requires a rank-2 tensor"
    );
    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    let entries: Vec<(Vec<isize>, T)> = (0..rows * cols)
        .into_par_iter()
        .map(|k| {
            let i = k / cols;
            let j = k % cols;
            (
                vec![j as isize, i as isize],
                tensor.get(&[i as isize, j as isize]).conj(),
            )
        })
        .collect();

    let mut out = A::empty(&[cols, rows]);
    for (idx, value) in entries {
        out.set(&idx, value);
    }
    out
}

/// Type-preserving dot product without conjugation.
pub fn dot<T, Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> T
where
    T: Scalar + Copy + Send + Sync,
    Lhs: TensorTrait<T>,
    Rhs: TensorTrait<T>,
{
    assert_same_shape::<T, _, _>(lhs, rhs);
    (0..lhs.size())
        .into_par_iter()
        .map(|k| {
            let idx = flat_to_index(lhs.shape(), k);
            lhs.get(&idx) * rhs.get(&idx)
        })
        .reduce(|| T::zero(), |a, b| a + b)
}

/// Type-preserving Hermitian dot product: `sum(conj(lhs[i]) * rhs[i])`.
pub fn hermitian_dot<T, Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> T
where
    T: Scalar + Copy + Send + Sync,
    Lhs: TensorTrait<T>,
    Rhs: TensorTrait<T>,
{
    assert_same_shape::<T, _, _>(lhs, rhs);
    (0..lhs.size())
        .into_par_iter()
        .map(|k| {
            let idx = flat_to_index(lhs.shape(), k);
            lhs.get(&idx).conj() * rhs.get(&idx)
        })
        .reduce(|| T::zero(), |a, b| a + b)
}

/// Real-valued squared norm projection.
pub fn norm_sqr_real<T, A>(tensor: &A) -> T::Real
where
    T: Scalar + Copy + Send + Sync,
    T::Real: Send + Sync,
    A: TensorTrait<T>,
{
    (0..tensor.size())
        .into_par_iter()
        .map(|k| {
            let idx = flat_to_index(tensor.shape(), k);
            tensor.get(&idx).norm_sqr_real()
        })
        .reduce(|| T::Real::zero(), |a, b| a + b)
}

/// Type-preserving Euclidean norm: `sqrt(sum(norm_sqr_real(x_i)))`.
pub fn norm<T, A>(tensor: &A) -> T
where
    T: Scalar + Copy + Send + Sync,
    T::Real: Send + Sync,
    A: TensorTrait<T>,
{
    T::from_re_im(norm_sqr_real::<T, A>(tensor).sqrt(), T::Real::zero())
}

/// Type-preserving 3D vector cross product.
pub fn cross<T, Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> Lhs
where
    T: Scalar + Copy + Send + Sync,
    Lhs: TensorTrait<T>,
    Rhs: TensorTrait<T>,
{
    assert_vector_len(lhs, 3, "cross");
    assert_vector_len(rhs, 3, "cross");

    let a0 = lhs.get(&[0]);
    let a1 = lhs.get(&[1]);
    let a2 = lhs.get(&[2]);
    let b0 = rhs.get(&[0]);
    let b1 = rhs.get(&[1]);
    let b2 = rhs.get(&[2]);

    let mut out = Lhs::empty(&[3]);
    let values: Vec<(usize, T)> = (0..3)
        .into_par_iter()
        .map(|i| {
            let value = match i {
                0 => a1 * b2 - a2 * b1,
                1 => a2 * b0 - a0 * b2,
                _ => a0 * b1 - a1 * b0,
            };
            (i, value)
        })
        .collect();
    for (i, value) in values {
        out.set(&[i as isize], value);
    }
    out
}

/// Type-preserving exterior product of two vectors.
///
/// Returns the antisymmetric rank-2 tensor `a_i b_j - a_j b_i`.
pub fn wedge<T, Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> Lhs
where
    T: Scalar + Copy + Send + Sync,
    Lhs: TensorTrait<T>,
    Rhs: TensorTrait<T>,
{
    assert_vector(lhs, "wedge");
    assert_vector(rhs, "wedge");
    let n = lhs.shape()[0];
    assert_eq!(rhs.shape()[0], n, "wedge vector length mismatch");

    let entries: Vec<(Vec<isize>, T)> = (0..n * n)
        .into_par_iter()
        .map(|k| {
            let i = k / n;
            let j = k % n;
            let value = lhs.get(&[i as isize]) * rhs.get(&[j as isize])
                - lhs.get(&[j as isize]) * rhs.get(&[i as isize]);
            (vec![i as isize, j as isize], value)
        })
        .collect();

    let mut out = Lhs::empty(&[n, n]);
    for (idx, value) in entries {
        out.set(&idx, value);
    }
    out
}

/// Type-preserving rank-2 matrix multiplication.
pub fn matmul<T, Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> Lhs
where
    T: Scalar + Copy + Send + Sync,
    Lhs: TensorTrait<T>,
    Rhs: TensorTrait<T>,
{
    assert_eq!(lhs.rank(), 2, "matmul lhs must be rank 2");
    assert_eq!(rhs.rank(), 2, "matmul rhs must be rank 2");
    let rows = lhs.shape()[0];
    let inner = lhs.shape()[1];
    assert_eq!(rhs.shape()[0], inner, "matmul inner dimensions mismatch");
    let cols = rhs.shape()[1];

    let entries: Vec<(Vec<isize>, T)> = (0..rows * cols)
        .into_par_iter()
        .map(|k| {
            let i = k / cols;
            let j = k % cols;
            let value = (0..inner)
                .into_par_iter()
                .map(|m| lhs.get(&[i as isize, m as isize]) * rhs.get(&[m as isize, j as isize]))
                .reduce(|| T::zero(), |a, b| a + b);
            (vec![i as isize, j as isize], value)
        })
        .collect();

    let mut out = Lhs::empty(&[rows, cols]);
    for (idx, value) in entries {
        out.set(&idx, value);
    }
    out
}

fn flat_to_index(shape: &[usize], mut flat: usize) -> Vec<isize> {
    let mut idx = vec![0isize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        idx[axis] = (flat % dim) as isize;
        flat /= dim;
    }
    idx
}

fn assert_vector<T, A>(tensor: &A, op: &str)
where
    T: Scalar,
    A: TensorTrait<T>,
{
    assert_eq!(tensor.rank(), 1, "{op} requires rank-1 vectors");
}

fn assert_vector_len<T, A>(tensor: &A, len: usize, op: &str)
where
    T: Scalar,
    A: TensorTrait<T>,
{
    assert_vector::<T, A>(tensor, op);
    assert_eq!(tensor.shape()[0], len, "{op} requires length-{len} vectors");
}

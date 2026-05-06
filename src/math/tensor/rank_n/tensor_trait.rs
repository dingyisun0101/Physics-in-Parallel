// src/math_foundations/tensor/tensor_trait.rs
/*!
A self-contained tensor contract for math-facing code.

`TensorTrait<T>` is the common way to interact with tensors without caring about
the backend memory layout. Dense and sparse tensors can use different storage
and traversal algorithms, but users should see the same mathematical object:
    a shaped collection of scalar values.

Design contract
- Tensor elements are always `T: Scalar`.
- Tensor math is expressed through scalar behavior. Operators such as `+` and
  `*` are valid only through generic scalar bounds; scalar-specific operations
  such as conjugation, absolute value, norm, square root, and casting call the
  `Scalar` trait directly.
- Type-preserving operations return `Self` or `T`. Operations that project out
  of the scalar category use explicit `_real` names.
- Tensor-wide maps, zips, reductions, fills, and casts should use Rayon where
  parallel traversal does not change observable semantics.
- Backend implementations may optimize storage traversal, but they must not
  change the mathematical meaning of the user-facing methods.

Indexing contract
- Multi-indices use `isize` so negative indices are available.
- Current dense and sparse backends normalize indices periodically. For shape
  `[n0, n1, ...]`, an axis index `a` is mapped modulo that axis length.
- `get` returns by value. Sparse tensors synthesize `T::zero()` for implicit
  zeros.
- `get_mut` returns a mutable value at the normalized index. Sparse tensors
  materialize an implicit zero if needed; callers can prune explicit zeros with
  backend-specific APIs after direct mutation.

Sparse semantics
- Methods whose name starts with `par_` operate according to backend traversal
  rules. Dense backends visit all logical elements. Sparse backends generally
  visit stored nonzeros and prune zeros produced by the operation.
- Fully mathematical sparse operations that would densify implicit zeros should
  either return a dense representation or use an explicit method name in later
  API layers.
*/

use crate::math::scalar::Scalar;

use super::ops;

/// Generic tensor behavior implemented by dense and sparse tensor backends.
pub trait TensorTrait<T: Scalar>: Send + Sync + Clone {
    /// Backend-specific representation returned when casting to `U`.
    ///
    /// Dense tensors should use dense output. Sparse tensors should use sparse
    /// output unless an operation explicitly documents densification.
    type Repr<U: Scalar>: TensorTrait<U>;

    // ---------------------------------------------------------------------
    // Required backend primitives
    // ---------------------------------------------------------------------

    /// Create a tensor with this backend's empty value for `shape`.
    fn empty(shape: &[usize]) -> Self;

    /// Shape metadata. The length of this slice is the tensor rank.
    fn shape(&self) -> &[usize];

    /// Row-major flat index for a normalized backend index policy.
    fn index(&self, idx: &[isize]) -> usize;

    /// Get by value at `idx`.
    fn get(&self, idx: &[isize]) -> T
    where
        T: Copy;

    /// Mutable reference at `idx`.
    ///
    /// Sparse backends may materialize an implicit zero to satisfy this method.
    fn get_mut(&mut self, idx: &[isize]) -> &mut T;

    /// Set the value at `idx`.
    ///
    /// Sparse backends should remove explicit storage when `val == T::zero()`.
    fn set(&mut self, idx: &[isize], val: T);

    /// Backend-native sum reduction.
    fn get_sum(&self) -> T;

    /// Backend-native parallel fill.
    fn par_fill(&mut self, value: T)
    where
        T: Copy + Send + Sync;

    /// Backend-native parallel in-place map: `x <- f(x)`.
    fn par_map_in_place<F>(&mut self, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send;

    /// Backend-native parallel in-place zip:
    ///	`self[i] <- f(self[i], other[i])`.
    fn par_zip_with_inplace<F, Rhs>(&mut self, other: &Rhs, f: F)
    where
        Rhs: TensorTrait<T>,
        T: Copy + Send + Sync,
        F: Fn(T, T) -> T + Sync + Send;

    /// Cast element type to another scalar `U`.
    fn cast_to<U: Scalar + Send + Sync>(&self) -> Self::Repr<U>
    where
        T: Copy + Send + Sync;

    /// Print a human-readable representation.
    fn print(&self);

    // ---------------------------------------------------------------------
    // Self-contained user-facing helpers
    // ---------------------------------------------------------------------

    /// Tensor rank, equal to `shape().len()`.
    #[inline]
    fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Logical dense size, equal to the product of all shape axes.
    #[inline]
    fn size(&self) -> usize {
        ops::size(self.shape())
    }

    /// Alias for `get_sum`.
    #[inline]
    fn sum(&self) -> T {
        self.get_sum()
    }

    /// Return a mapped tensor with the same backend.
    #[inline]
    fn map<F>(&self, f: F) -> Self
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send,
    {
        ops::map::<T, Self, F>(self, f)
    }

    /// Map this tensor in place.
    #[inline]
    fn map_in_place<F>(&mut self, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send,
    {
        self.par_map_in_place(f);
    }

    /// Return a tensor produced by zipping two tensors of the same shape.
    #[inline]
    fn zip_with<Rhs, F>(&self, rhs: &Rhs, f: F) -> Self
    where
        Rhs: TensorTrait<T>,
        T: Copy + Send + Sync,
        F: Fn(T, T) -> T + Sync + Send,
    {
        ops::zip_with::<T, Self, Rhs, F>(self, rhs, f)
    }

    /// Fill according to backend traversal semantics.
    #[inline]
    fn fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.par_fill(value);
    }

    /// Type-preserving elementwise conjugate.
    #[inline]
    fn conj(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        ops::conj::<T, Self>(self)
    }

    /// Type-preserving elementwise absolute value.
    #[inline]
    fn abs(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        ops::abs::<T, Self>(self)
    }

    /// Type-preserving elementwise squared norm.
    #[inline]
    fn norm_sqr(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        ops::norm_sqr::<T, Self>(self)
    }

    /// Type-preserving elementwise square root.
    #[inline]
    fn sqrt(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        ops::sqrt::<T, Self>(self)
    }

    /// Type-preserving scalar multiplication.
    #[inline]
    fn scalar_mul(&self, scalar: T) -> Self
    where
        T: Copy + Send + Sync,
    {
        ops::scalar_mul::<T, Self>(self, scalar)
    }

    /// Type-preserving elementwise tensor multiplication.
    #[inline]
    fn elem_mul<Rhs>(&self, rhs: &Rhs) -> Self
    where
        Rhs: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        ops::elem_mul::<T, Self, Rhs>(self, rhs)
    }

    /// Type-preserving elementwise tensor division.
    #[inline]
    fn elem_div<Rhs>(&self, rhs: &Rhs) -> Self
    where
        Rhs: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        ops::elem_div::<T, Self, Rhs>(self, rhs)
    }

    /// Type-preserving rank-2 transpose.
    #[inline]
    fn transpose(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        ops::transpose::<T, Self>(self)
    }

    /// Type-preserving rank-2 Hermitian transpose.
    #[inline]
    fn hermitian_transpose(&self) -> Self
    where
        T: Copy + Send + Sync,
    {
        ops::hermitian_transpose::<T, Self>(self)
    }

    /// Type-preserving dot product without conjugation.
    #[inline]
    fn dot<Rhs>(&self, rhs: &Rhs) -> T
    where
        Rhs: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        ops::dot::<T, Self, Rhs>(self, rhs)
    }

    /// Type-preserving Hermitian dot product: `sum(conj(self[i]) * rhs[i])`.
    #[inline]
    fn hermitian_dot<Rhs>(&self, rhs: &Rhs) -> T
    where
        Rhs: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        ops::hermitian_dot::<T, Self, Rhs>(self, rhs)
    }

    /// Real-valued squared norm projection.
    #[inline]
    fn norm_sqr_real(&self) -> T::Real
    where
        T: Copy + Send + Sync,
        T::Real: Send + Sync,
    {
        ops::norm_sqr_real::<T, Self>(self)
    }

    /// Type-preserving Euclidean norm.
    #[inline]
    fn norm(&self) -> T
    where
        T: Copy + Send + Sync,
        T::Real: Send + Sync,
    {
        ops::norm::<T, Self>(self)
    }

    /// Type-preserving 3D vector cross product.
    #[inline]
    fn cross<Rhs>(&self, rhs: &Rhs) -> Self
    where
        Rhs: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        ops::cross::<T, Self, Rhs>(self, rhs)
    }

    /// Type-preserving exterior product of two vectors.
    #[inline]
    fn wedge<Rhs>(&self, rhs: &Rhs) -> Self
    where
        Rhs: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        ops::wedge::<T, Self, Rhs>(self, rhs)
    }

    /// Type-preserving rank-2 matrix multiplication.
    #[inline]
    fn matmul<Rhs>(&self, rhs: &Rhs) -> Self
    where
        Rhs: TensorTrait<T>,
        T: Copy + Send + Sync,
    {
        ops::matmul::<T, Self, Rhs>(self, rhs)
    }
}

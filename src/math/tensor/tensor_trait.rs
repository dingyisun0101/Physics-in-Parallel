// src/math_foundations/tensor/tensor_trait.rs
/*!
A **unified tensor trait** shared by both dense and sparse backends.

This trait defines a common API surface so your `dense::Tensor<T>` and
`sparse::Tensor<T>` can be used interchangeably by generic code.

Key points
- Multi-indexing accepts `isize` per axis with negative indices permitted
  (backend is responsible for normalization).
- `get` returns the value **by copy** (sparse backends synthesize zero when
  the entry is not explicitly stored).
- `get_mut` returns `Option<&mut T>` (sparse returns `None` for implicit zeros).
- Parallel ops follow backend semantics:
  - `par_fill`: dense → all elements; sparse → existing nonzeros only.
  - `par_map_in_place`: in-place map; sparse maps only stored nonzeros and prunes
    zeros produced by the map.
  - `par_zip_with_inplace`: combine **self’s elements** with `other` via `f`;
    sparse applies only where `self` currently stores nonzeros and prunes zeros.
- `cast_to<U>` converts the element type; each backend returns its natural
  representation via the associated type `Repr<U>` (dense→dense, sparse→sparse).
*/

use super::super::scalar::Scalar;


/// Unified tensor behavior (to be implemented by dense and sparse tensors).
pub trait TensorTrait<T: Scalar>: Send + Sync {
    /// Backend-specific representation returned when casting to `U`.
    ///
    /// - Dense implementation should set `Repr<U> = dense::Tensor<U>`.
    /// - Sparse implementation should set `Repr<U> = sparse::Tensor<U>`.
    type Repr<U: Scalar>: TensorTrait<U>;

    /// Shape vector.
    fn shape(&self) -> &[usize];

    // ------------------------ Indexing & Element Access ------------------------

    /// Row-major flat index for a possibly-negative per-axis index.
    fn index(&self, idx: &[isize]) -> usize;

    /// Get **by value** at `idx`. (Sparse backends return zero for missing.)
    fn get(&self, idx: &[isize]) -> T
    where
        T: Copy;

    /// Mutable reference to the element at `idx` **if it exists**.
    ///
    /// - Dense: always `Some(&mut T)`.
    /// - Sparse: `Some(&mut T)` only for stored nonzeros; `None` for implicit zeros.
    fn get_mut(&mut self, idx: &[isize]) -> Option<&mut T>;

    /// Set the value at `idx`. Writing zero may prune storage in sparse backends.
    fn set(&mut self, idx: &[isize], val: T);

    // ----------------------------- Parallel Ops -------------------------------

    /// Parallel fill.
    ///
    /// - Dense: set **every element** to `value`.
    /// - Sparse: set **existing nonzeros** to `value` (and clear if `value == 0`).
    fn par_fill(&mut self, value: T)
    where
        T: Copy + Send + Sync;

    /// Parallel in-place elementwise map: `x <- f(x)`.
    ///
    /// - Dense: applies to all elements.
    /// - Sparse: applies to stored nonzeros; results equal to zero are pruned.
    fn par_map_in_place<F>(&mut self, f: F)
    where
        T: Copy + Send + Sync,
        F: Fn(T) -> T + Sync + Send;

    /// Parallel zip-with (in place on `self`): `self[i] <- f(self[i], other[i])`.
    ///
    /// - Dense: applies to all elements.
    /// - Sparse: applies only over **self’s nonzeros** (zeros produced are pruned).
    fn par_zip_with_inplace<F, Rhs>(&mut self, other: &Rhs, f: F)
    where
        Rhs: TensorTrait<T> + ?Sized,
        T: Copy + Send + Sync,
        F: Fn(T, T) -> T + Sync + Send;

    // ----------------------------- Type Casting -------------------------------

    /// Cast element type to another scalar `U`. Backend chooses representation via `Repr<U>`.
    fn cast_to<U: Scalar + Send + Sync>(&self) -> Self::Repr<U>
    where
        T: Copy + Send + Sync;
}

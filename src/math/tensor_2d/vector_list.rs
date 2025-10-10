use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use rayon::prelude::*;

use crate::math::scalar::Scalar;
use crate::math::tensor::{dense::Tensor, tensor_trait::TensorTrait};
use crate::math::tensor_2d::matrix::{Matrix, RowRef, RowRefMut, ColRef, ColRefMut};

// ============================================================================
// ------------------------------- Core Structs -------------------------------
// ============================================================================
#[derive(Debug, Clone)]
pub struct VectorList<T: Scalar, B: TensorTrait<T> = Tensor<T>> {
    pub matrix: Matrix<T, B>, // rows = dim (D), cols = n
    _pd: PhantomData<T>,
}




// ============================================================================
// --------------------------------- Basic I/O --------------------------------
// ============================================================================

impl<T: Scalar, B: TensorTrait<T>> VectorList<T, B> {
    /// New `[dim, n]` SoA container.
    #[inline]
    pub fn empty(dim: usize, n: usize) -> Self {
        assert!(dim > 0 && n > 0, "VectorList::new: shape must be nonzero");
        Self {
            matrix: Matrix::<T, B>::empty(dim, n),
            _pd: PhantomData,
        }
    }

    /// Wrap an existing matrix (must be rank-2).
    #[inline]
    pub fn from_matrix(m: Matrix<T, B>) -> Self {
        // trust Matrix invariants
        Self { matrix: m, _pd: PhantomData }
    }

    /// Extract the backing `Matrix`.
    #[inline]
    pub fn into_matrix(self) -> Matrix<T, B> { self.matrix }

    // ----------------------- shape helpers -----------------------
    #[inline] pub fn dim(&self) -> usize { self.matrix.rows() }
    #[inline] pub fn num_vectors(&self) -> usize { self.matrix.cols() }
    #[inline] pub fn shape(&self) -> (usize, usize) { (self.dim(), self.num_vectors()) }

    // ------------------------ Type helpers ------------------------
    #[inline]
    pub fn cast_to<U>(&self) -> VectorList<U, B::Repr<U>>
    where
        T: Copy + Send + Sync,
        U: Scalar + Send + Sync,
    {
        let data_u: Matrix<U, B::Repr<U>> = self.matrix.cast_to::<U>();
        VectorList { matrix: data_u, _pd: PhantomData }
    }
}




// ============================================================================
// --------------------------------- Accesors ---------------------------------
// ============================================================================

impl<T: Scalar, B: TensorTrait<T>> VectorList<T, B> {
    // --------------------- element accessors ---------------------
    /// Get by value at vector `i` (column), axis `k` (row).
    #[inline]
    pub fn get(&self, i: isize, k: isize) -> T where T: Copy {
        self.matrix.get(k, i)
    }

    /// Set at vector `i` (column), axis `k` (row).
    #[inline]
    pub fn set(&mut self, i: isize, k: isize, val: T) {
        self.matrix.set(k, i, val);
    }

    /// Mutable ref to entry `(i, k)` (vector `i`, axis `k`).
    #[inline]
    pub fn get_mut(&mut self, i: isize, k: isize) -> &mut T {
        self.matrix.get_mut(k, i)
    }

    // ------------------------- views -----------------------------
    /// Axis view (row) for dimension `k`.
    #[inline] pub fn get_axis(&self, k: isize) -> RowRef<'_, T, B> { self.matrix.row(k) }
    #[inline] pub fn get_axis_mut(&mut self, k: isize) -> RowRefMut<'_, T, B> { self.matrix.row_mut(k) }

    /// Vector view (column) for vector `i`.
    #[inline] pub fn get_vector(&self, i: isize) -> ColRef<'_, T, B> { self.matrix.col(i) }
    #[inline] pub fn get_vector_mut(&mut self, i: isize) -> ColRefMut<'_, T, B> { self.matrix.col_mut(i) }
}






// ============================================================================
// ---------------------------- Bulk Operations -------------------------------
// ============================================================================
/// Scaling
impl<T, B> VectorList<T, B>
where
    T: Scalar + Copy + Send + Sync + Mul<Output = T>,
    B: TensorTrait<T> + Send + Sync,
{
    /// Scale each **vector (column)** by the corresponding factor in `scales`.
    ///
    /// - `scales.len()` must equal `self.num_vectors()`.
    /// - Parallelized across **rows** to compute the scaled values, then
    ///   committed back in a single pass. (No auxiliary `VectorList` is created.)
    #[inline]
    pub fn scale_vectors_by_list(&mut self, scales: &[T]) {
        let d = self.dim();
        let n = self.num_vectors();
        assert!(
            scales.len() == n,
            "scale_vectors_by_list: len mismatch (got {}, expected {})",
            scales.len(),
            n
        );

        // Phase 1 (parallel): compute scaled rows
        // Each worker reads from &self (immutable) and writes to its own buffer.
        let scaled_rows: Vec<Vec<T>> = (0..d)
            .into_par_iter()
            .map(|k| {
                // build scaled row k: v_kj * scales[j]
                let mut out = Vec::with_capacity(n);
                for j in 0..n {
                    let x = self.matrix.get(k as isize, j as isize);
                    out.push(x * scales[j]);
                }
                out
            })
            .collect();

        // Phase 2 (sequential): commit back in-place, row by row.
        for (k, row_vals) in scaled_rows.into_iter().enumerate() {
            self.matrix.row_mut(k as isize).set_from_slice(&row_vals);
        }
    }

    /// Parallel scale: v[:, j] <- s * v[:, j] for all j
    #[inline]
    pub fn scale_vectors_by_scalar(&mut self, s: T) {
        self.matrix.as_tensor_mut().par_map_in_place(|x| x * s);
    }
}


/// Bulk set operations
impl<T: Scalar, B: TensorTrait<T>> VectorList<T, B> {
    /// Set an entire **vector** (column) `i` from slice `vals` of length `dim()`.
    #[inline]
    pub fn set_vector_from_slice(&mut self, i: isize, vals: &[T])
    where
        T: Copy,
    {
        assert!(vals.len() == self.dim(), "set_vector_from_slice: len mismatch");
        self.matrix.col_mut(i).set_from_slice(vals);
    }

    /// Set an entire **axis** (row) `k` from slice `vals` of length `num_vectors()`.
    #[inline]
    pub fn set_axis_from_slice(&mut self, k: isize, vals: &[T])
    where
        T: Copy,
    {
        assert!(vals.len() == self.num_vectors(), "set_axis_from_slice: len mismatch");
        self.matrix.row_mut(k).set_from_slice(vals);
    }
}



// ============================================================================
// ------------------------------ Math Utils ----------------------------------
// ============================================================================

/// Norms
impl<T, B> VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T> + Send + Sync,
{
    /// Return the L2 norms of each **vector (column)** as `Vec<T>`.
    /// Parallelized across columns with Rayon.
    #[inline]
    pub fn get_norms(&self) -> Vec<T> {
        let d = self.dim();
        let n = self.num_vectors();

        (0..n).into_par_iter()
            .map(|j| {
                let mut s = T::zero();
                for k in 0..d {
                    let x = self.matrix.get(k as isize, j as isize);
                    s = s + x * x;
                }
                <T>::sqrt(s)
            })
            .collect()
    }

    pub fn normalize(&mut self) {
        let norms = self.get_norms();
        self.scale_vectors_by_list(&norms.iter().map(|&x| T::one() / x).collect::<Vec<T>>());
    }
}



// ============================================================================
// --------------------------- Arithmetic Ops ---------------------------------
// ============================================================================

// -------------------- &VectorList ⊕ &VectorList -> VectorList --------------------

impl<'a, T, B> Add<&'a VectorList<T, B>> for &'a VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Add<&'b B, Output = B>,
{
    type Output = VectorList<T, B>;
    #[inline]
    fn add(self, rhs: &'a VectorList<T, B>) -> Self::Output {
        VectorList { matrix: &self.matrix + &rhs.matrix, _pd: PhantomData }
    }
}

impl<'a, T, B> Sub<&'a VectorList<T, B>> for &'a VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Sub<&'b B, Output = B>,
{
    type Output = VectorList<T, B>;
    #[inline]
    fn sub(self, rhs: &'a VectorList<T, B>) -> Self::Output {
        VectorList { matrix: &self.matrix - &rhs.matrix, _pd: PhantomData }
    }
}

impl<'a, T, B> Mul<&'a VectorList<T, B>> for &'a VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Mul<&'b B, Output = B>,
{
    type Output = VectorList<T, B>;
    #[inline]
    fn mul(self, rhs: &'a VectorList<T, B>) -> Self::Output {
        VectorList { matrix: &self.matrix * &rhs.matrix, _pd: PhantomData }
    }
}

impl<'a, T, B> Div<&'a VectorList<T, B>> for &'a VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Div<&'b B, Output = B>,
{
    type Output = VectorList<T, B>;
    #[inline]
    fn div(self, rhs: &'a VectorList<T, B>) -> Self::Output {
        VectorList { matrix: &self.matrix / &rhs.matrix, _pd: PhantomData }
    }
}

// -------------------- VectorList op= VectorList --------------------

impl<T, B> AddAssign<&VectorList<T, B>> for VectorList<T, B>
where
    T: Scalar + Copy + Send + Sync + core::ops::Add<Output = T>,
    B: TensorTrait<T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: &VectorList<T, B>) {
        self.matrix += &rhs.matrix;
    }
}

impl<T, B> SubAssign<&VectorList<T, B>> for VectorList<T, B>
where
    T: Scalar + Copy + Send + Sync + core::ops::Sub<Output = T>,
    B: TensorTrait<T>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &VectorList<T, B>) {
        self.matrix -= &rhs.matrix;
    }
}

impl<T, B> MulAssign<&VectorList<T, B>> for VectorList<T, B>
where
    T: Scalar + Copy + Send + Sync + core::ops::Mul<Output = T>,
    B: TensorTrait<T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &VectorList<T, B>) {
        self.matrix *= &rhs.matrix;
    }
}

impl<T, B> DivAssign<&VectorList<T, B>> for VectorList<T, B>
where
    T: Scalar + Copy + Send + Sync + core::ops::Div<Output = T>,
    B: TensorTrait<T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: &VectorList<T, B>) {
        self.matrix /= &rhs.matrix;
    }
}

// -------------------- &VectorList ⊕ scalar -> VectorList --------------------

impl<'a, T, B> Add<T> for &'a VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Add<T, Output = B>,
{
    type Output = VectorList<T, B>;
    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        VectorList { matrix: &self.matrix + rhs, _pd: PhantomData }
    }
}

impl<'a, T, B> Sub<T> for &'a VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Sub<T, Output = B>,
{
    type Output = VectorList<T, B>;
    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        VectorList { matrix: &self.matrix - rhs, _pd: PhantomData }
    }
}

impl<'a, T, B> Mul<T> for &'a VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Mul<T, Output = B>,
{
    type Output = VectorList<T, B>;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        VectorList { matrix: &self.matrix * rhs, _pd: PhantomData }
    }
}

impl<'a, T, B> Div<T> for &'a VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T>,
    for<'b> &'b B: Div<T, Output = B>,
{
    type Output = VectorList<T, B>;
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        VectorList { matrix: &self.matrix / rhs, _pd: PhantomData }
    }
}

// -------------------- VectorList scalar-op --------------------

impl<T, B> AddAssign<T> for VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T> + AddAssign<T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.matrix += rhs;
    }
}

impl<T, B> SubAssign<T> for VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T> + SubAssign<T>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.matrix -= rhs;
    }
}

impl<T, B> MulAssign<T> for VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T> + MulAssign<T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.matrix *= rhs;
    }
}

impl<T, B> DivAssign<T> for VectorList<T, B>
where
    T: Scalar,
    B: TensorTrait<T> + DivAssign<T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.matrix /= rhs;
    }
}

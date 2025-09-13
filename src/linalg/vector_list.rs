// src/linalg/vector_list.rs

use std::array::from_fn;
use num_traits::Float;
use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg};

use serde::{Serialize, Deserialize, Serializer, Deserializer, de};
use serde::de::DeserializeOwned;
use serde_json;

use rayon::prelude::*;




/// ================================================================================
/// ==========================  Vector List: SoA Layout ============================
/// ================================================================================

#[derive(Debug, Clone)]
pub struct VectorList<T, const D: usize> {
    // data[d][i] = i-th point’s coordinate in dimension d
    data: Vec<Vec<T>>,
}


/// Basic Impl
#[allow(dead_code)]
impl<T, const D: usize> VectorList<T, D> {
    /// Create empty (N=0), with D dimensions.
    pub fn new() -> Self {
        Self { data: (0..D).map(|_| Vec::new()).collect() }
    }

    /// Create with capacity for `n` points per dimension.
    pub fn with_capacity(n: usize) -> Self {
        Self { data: (0..D).map(|_| Vec::with_capacity(n)).collect() }
    }

    /// Number of dimensions (compile-time constant).
    #[inline] pub const fn dims(&self) -> usize { D }

    /// Number of points (N). All inner vecs have equal length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.first().map(|v| v.len()).unwrap_or(0)
    }

    #[inline] pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Immutable view of one dimension `d` (0 ≤ d < D).
    #[inline]
    pub fn dim(&self, d: usize) -> &[T] {
        &self.data[d]
    }

    /// Mutable view of one dimension `d` (0 ≤ d < D).
    #[inline]
    pub fn dim_mut(&mut self, d: usize) -> &mut Vec<T> {
        &mut self.data[d]
    }

    /// Push one point given as an array `[T; D]` (SoA append).
    pub fn push_point(&mut self, point: [T; D]) {
        for (d, val) in point.into_iter().enumerate() {
            self.data[d].push(val);
        }
    }

    /// Reserve additional capacity for `more` points in each dimension.
    pub fn reserve(&mut self, more: usize) {
        for dim in &mut self.data {
            dim.reserve(more);
        }
    }

    /// Get i-th point as an array. Requires `T: Clone` (or use `T: Copy`).
    pub fn point(&self, i: usize) -> [T; D]
    where
        T: Clone,
    {
        from_fn(|d| self.data[d][i].clone())
    }

    /// Convert from AoS: `Vec<[T; D]>` → SoA `VectorList`.
    pub fn from_points(points: Vec<[T; D]>) -> Self {
        let n = points.len();
        let mut out = Self::with_capacity(n);
        for p in points {
            out.push_point(p);
        }
        out
    }

    /// Convert to AoS: SoA `VectorList` → `Vec<[T; D]>`.
    pub fn to_points(&self) -> Vec<[T; D]>
    where
        T: Clone,
    {
        let n = self.len();
        (0..n).map(|i| self.point(i)).collect()
    }

    /// Validate internal invariants (all dims same length).
    pub fn validate(&self) -> bool {
        self.data.iter().all(|v| v.len() == self.len()) && self.data.len() == D
    }

    /// Construct from `Vec<Vec<T>>` where `outer.len()==D` and all inner vecs equal length.
    pub fn try_from_vecvec(data: Vec<Vec<T>>) -> Result<Self, &'static str> {
        if data.len() != D { return Err("outer Vec length must equal D"); }
        let n = data[0].len();
        if !data.iter().all(|v| v.len() == n) {
            return Err("all inner Vecs must have the same length");
        }
        Ok(Self { data })
    }

    /// Expose raw data if you need direct control.
    #[inline] pub fn as_inner(&self) -> &Vec<Vec<T>> { &self.data }
    #[inline] pub fn into_inner(self) -> Vec<Vec<T>> { self.data }
}




/// ================================================================================
/// ==============================  Raw Accessors ==================================
/// ================================================================================
#[allow(dead_code)]
impl<T, const D: usize> VectorList<T, D> {
    /// Immutable element access: value at dimension `d`, element index `n`.
    #[inline]
    pub fn get(&self, d: usize, n: usize) -> Option<&T> {
        self.data.get(d)?.get(n)
    }

    /// Mutable element access.
    #[inline]
    pub fn get_mut(&mut self, d: usize, n: usize) -> Option<&mut T> {
        self.data.get_mut(d)?.get_mut(n)
    }

    /// Set element at (d, n) to a new value. Returns Err if out of bounds.
    #[inline]
    pub fn set(&mut self, d: usize, n: usize, val: T) -> Result<(), &'static str> {
        if let Some(dim) = self.data.get_mut(d) {
            if let Some(slot) = dim.get_mut(n) {
                *slot = val;
                return Ok(());
            }
        }
        Err("out of bounds")
    }
}


/// ================================================================================
/// ==========================  High-Level Accessors ===============================
/// ================================================================================
#[allow(dead_code)]
impl<T: Clone, const D: usize> VectorList<T, D> {
    // ---------------- High-level Accessors ----------------

    /// Return the n-th point as `[T; D]`. Needs `T: Clone`.
    #[inline]
    pub fn get_vec(&self, n: usize) -> Option<[T; D]>
    where
        T: Clone,
    {
        if n >= self.len() {
            return None;
        }
        Some(from_fn(|d| self.data[d][n].clone()))
    }

    /// Mutable reference to the n-th point as `[&mut T; D]`.
    #[inline]
    pub fn get_mut_vec(&mut self, n: usize) -> Option<[&mut T; D]> {
        if n >= self.len() {
            return None;
        }

        // Collect raw pointers for each dimension
        let mut ptrs: [*mut T; D] = from_fn(|d| {
            // Safe: n < len checked above
            let dim = &mut self.data[d];
            dim.as_mut_ptr().wrapping_add(n)
        });

        // SAFETY: each pointer points to a distinct Vec (no aliasing across dimensions)
        Some(unsafe { from_fn(|d| &mut *ptrs[d]) })
    }

    /// Overwrite the n-th point with a new vector `[T; D]`.
    #[inline]
    pub fn set_vec(&mut self, n: usize, val: [T; D]) -> Result<(), &'static str> {
        if n >= self.len() {
            return Err("out of bounds");
        }
        for d in 0..D {
            self.data[d][n] = val[d].clone();
        }
        Ok(())
    }
}



/// ================================================================================
/// ============================  LinAlg Utilities =================================
/// ================================================================================

#[allow(dead_code)]
impl<T, const D: usize> VectorList<T, D>
where
    T: Float + Send + Sync + Copy,
{
    /// Return a new `VectorList` whose points are normalized (unit length).
    /// Zero-length points remain zeros.
    pub fn normalized(&self) -> VectorList<T, D> {
        let (_norms, units) = self.to_norm_and_unit_vec();
        units
    }

    /// Compute per-point Euclidean norms and the corresponding unit vectors.
    /// Returns: (norms: Vec<T>, unit_vectors: VectorList<T, D>)
    /// Zero-length points keep unit vector = 0 along all dims.
    pub fn to_norm_and_unit_vec(&self) -> (Vec<T>, VectorList<T, D>) {
        let n = self.len();

        // 1) Norms (parallel over points)
        let norms: Vec<T> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut sum = T::zero();
                for d in 0..D {
                    let v = self.data[d][i];
                    sum = sum + v * v;
                }
                sum.sqrt()
            })
            .collect();

        // 2) Unit vectors (parallel over indices per dimension)
        // For each dimension d, produce a Vec<T> of length n in parallel.
        let unit_data: Vec<Vec<T>> = (0..D)
            .map(|d| {
                // For fixed d, fill entries in parallel over i.
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let norm = norms[i];
                        if norm > T::zero() {
                            self.data[d][i] / norm
                        } else {
                            T::zero()
                        }
                    })
                    .collect()
            })
            .collect();

        (norms, VectorList { data: unit_data })
    }
}




// =========================== Arithmetic Ops (SoA) ===========================
//
// Elementwise +, -, *, / between two VectorList<T, D>, and with a scalar T.
// Also provides the corresponding *_assign ops and unary Neg.
//
// Parallelized with rayon over point indices for each dimension.
//
// Panics if the two VectorLists have different lengths (N) or dimensions D.
// ===========================================================================

impl<T, const D: usize> VectorList<T, D>
where
    T: Copy + Send + Sync,
{
    #[inline]
    fn assert_same_shape(&self, rhs: &Self) {
        debug_assert_eq!(self.data.len(), D);
        debug_assert_eq!(rhs.data.len(), D);
        let n = self.len();
        if rhs.len() != n {
            panic!("VectorList shape mismatch: lhs N={}, rhs N={}", n, rhs.len());
        }
    }

    #[inline]
    fn map_unary<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T + Sync + Send,
    {
        let n = self.len();
        let data = (0..D)
            .map(|d| {
                (0..n)
                    .into_par_iter()
                    .map(|i| f(self.data[d][i]))
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<_>>();
        Self { data }
    }

    #[inline]
    fn map_binary<F>(&self, rhs: &Self, f: F) -> Self
    where
        F: Fn(T, T) -> T + Sync + Send,
    {
        self.assert_same_shape(rhs);
        let n = self.len();
        let data = (0..D)
            .map(|d| {
                let a = &self.data[d];
                let b = &rhs.data[d];
                (0..n)
                    .into_par_iter()
                    .map(|i| f(a[i], b[i]))
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<_>>();
        Self { data }
    }

    #[inline]
    fn map_unary_in_place<F>(&mut self, f: F)
    where
        F: Fn(T) -> T + Sync + Send,
    {
        for d in 0..D {
            self.data[d]
                .par_iter_mut()
                .for_each(|x| *x = f(*x));
        }
    }

    #[inline]
    fn map_binary_in_place<F>(&mut self, rhs: &Self, f: F)
    where
        F: Fn(T, T) -> T + Sync + Send,
    {
        self.assert_same_shape(rhs);
        for d in 0..D {
            let rb = &rhs.data[d];
            self.data[d]
                .par_iter_mut()
                .zip(rb.par_iter().copied())
                .for_each(|(a, b)| *a = f(*a, b));
        }
    }
}

// ----------------------------- Unary -----------------------------

impl<T, const D: usize> Neg for &VectorList<T, D>
where
    T: Copy + Send + Sync + Neg<Output = T>,
{
    type Output = VectorList<T, D>;
    #[inline]
    fn neg(self) -> Self::Output {
        self.map_unary(|x| -x)
    }
}

// ------------------------- Vector ⊕ Vector ------------------------

impl<T, const D: usize> Add<&VectorList<T, D>> for &VectorList<T, D>
where
    T: Copy + Send + Sync + Add<Output = T>,
{
    type Output = VectorList<T, D>;
    #[inline]
    fn add(self, rhs: &VectorList<T, D>) -> Self::Output {
        self.map_binary(rhs, |a, b| a + b)
    }
}

impl<T, const D: usize> Sub<&VectorList<T, D>> for &VectorList<T, D>
where
    T: Copy + Send + Sync + Sub<Output = T>,
{
    type Output = VectorList<T, D>;
    #[inline]
    fn sub(self, rhs: &VectorList<T, D>) -> Self::Output {
        self.map_binary(rhs, |a, b| a - b)
    }
}

impl<T, const D: usize> Mul<&VectorList<T, D>> for &VectorList<T, D>
where
    T: Copy + Send + Sync + Mul<Output = T>,
{
    type Output = VectorList<T, D>;
    #[inline]
    fn mul(self, rhs: &VectorList<T, D>) -> Self::Output {
        self.map_binary(rhs, |a, b| a * b)
    }
}

impl<T, const D: usize> Div<&VectorList<T, D>> for &VectorList<T, D>
where
    T: Copy + Send + Sync + Div<Output = T>,
{
    type Output = VectorList<T, D>;
    #[inline]
    fn div(self, rhs: &VectorList<T, D>) -> Self::Output {
        self.map_binary(rhs, |a, b| a / b)
    }
}

// ------------------------- Vector ⊕ Scalar ------------------------

impl<T, const D: usize> Add<T> for &VectorList<T, D>
where
    T: Copy + Send + Sync + Add<Output = T>,
{
    type Output = VectorList<T, D>;
    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        self.map_unary(|x| x + rhs)
    }
}

impl<T, const D: usize> Sub<T> for &VectorList<T, D>
where
    T: Copy + Send + Sync + Sub<Output = T>,
{
    type Output = VectorList<T, D>;
    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        self.map_unary(|x| x - rhs)
    }
}

impl<T, const D: usize> Mul<T> for &VectorList<T, D>
where
    T: Copy + Send + Sync + Mul<Output = T>,
{
    type Output = VectorList<T, D>;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        self.map_unary(|x| x * rhs)
    }
}

impl<T, const D: usize> Div<T> for &VectorList<T, D>
where
    T: Copy + Send + Sync + Div<Output = T>,
{
    type Output = VectorList<T, D>;
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        self.map_unary(|x| x / rhs)
    }
}

// --------------------------- *_assign ops --------------------------

impl<T, const D: usize> AddAssign<&VectorList<T, D>> for VectorList<T, D>
where
    T: Copy + Send + Sync + Add<Output = T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: &VectorList<T, D>) {
        self.map_binary_in_place(rhs, |a, b| a + b);
    }
}

impl<T, const D: usize> SubAssign<&VectorList<T, D>> for VectorList<T, D>
where
    T: Copy + Send + Sync + Sub<Output = T>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &VectorList<T, D>) {
        self.map_binary_in_place(rhs, |a, b| a - b);
    }
}

impl<T, const D: usize> MulAssign<&VectorList<T, D>> for VectorList<T, D>
where
    T: Copy + Send + Sync + Mul<Output = T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &VectorList<T, D>) {
        self.map_binary_in_place(rhs, |a, b| a * b);
    }
}

impl<T, const D: usize> DivAssign<&VectorList<T, D>> for VectorList<T, D>
where
    T: Copy + Send + Sync + Div<Output = T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: &VectorList<T, D>) {
        self.map_binary_in_place(rhs, |a, b| a / b);
    }
}

impl<T, const D: usize> AddAssign<T> for VectorList<T, D>
where
    T: Copy + Send + Sync + Add<Output = T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.map_unary_in_place(|x| x + rhs);
    }
}

impl<T, const D: usize> SubAssign<T> for VectorList<T, D>
where
    T: Copy + Send + Sync + Sub<Output = T>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.map_unary_in_place(|x| x - rhs);
    }
}

impl<T, const D: usize> MulAssign<T> for VectorList<T, D>
where
    T: Copy + Send + Sync + Mul<Output = T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.map_unary_in_place(|x| x * rhs);
    }
}

impl<T, const D: usize> DivAssign<T> for VectorList<T, D>
where
    T: Copy + Send + Sync + Div<Output = T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.map_unary_in_place(|x| x / rhs);
    }
}





// ─────────────────────────────────────────────────────────────────────────────
// Serde support (JSON) for VectorList<T, D>
// - Serializes as a plain Vec<Vec<T>> in SoA layout: data[d][i]
// - During deserialization, validates outer.len() == D and all inner lengths equal
// - Convenience helpers: to_json_string / from_json_str
// Enable via Cargo feature `with-serde`
// ─────────────────────────────────────────────────────────────────────────────

impl<T, const D: usize> Serialize for VectorList<T, D>
where
    T: Serialize,
{
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as the inner SoA Vec<Vec<T>>
        self.data.serialize(serializer)
    }
}


impl<'de, T, const D: usize> Deserialize<'de> for VectorList<T, D>
where
    T: Deserialize<'de>,
{
    #[inline]
    fn deserialize<Dser>(deserializer: Dser) -> Result<Self, Dser::Error>
    where
        Dser: Deserializer<'de>,
    {
        // Expect a Vec<Vec<T>> in SoA form
        let data = <Vec<Vec<T>>>::deserialize(deserializer)?;

        if data.len() != D {
            return Err(de::Error::custom(format!(
                "VectorList: expected outer length D = {}, got {}",
                D, data.len()
            )));
        }
        let n = data[0].len();
        if !data.iter().all(|v| v.len() == n) {
            return Err(de::Error::custom(
                "VectorList: all inner Vecs must have the same length",
            ));
        }
        Ok(Self { data })
    }
}

impl<T, const D: usize> VectorList<T, D>
where
    T: Serialize + DeserializeOwned,
{
    /// Serialize to a compact JSON string (SoA `Vec<Vec<T>>`).
    #[inline]
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self)
    }

    /// Serialize to pretty JSON string.
    #[inline]
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self)
    }

    /// Deserialize from JSON string into `VectorList<T, D>`.
    #[inline]
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str::<Self>(s)
    }
}
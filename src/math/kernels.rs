//! Contiguous numerical kernels behind the universal containers.
//!
//! Optional x86 AVX-512F/AVX2 code is selected at runtime. Other targets and short
//! inputs use ordinary Rust loops, which may themselves be auto-vectorized.
//! Slices need no special alignment. Every kernel processes a scalar tail and
//! preserves separate multiplication/addition (no implicit FMA or fast math).

use crate::math::Scalar;

/// Internal elementwise operation, also used to select sparse support rules.
#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl BinaryOp {
    #[inline]
    pub(crate) fn apply<T: Scalar>(self, left: T, right: T) -> T {
        match self {
            Self::Add => left + right,
            Self::Subtract => left - right,
            Self::Multiply => left * right,
            Self::Divide => left / right,
        }
    }

    pub(crate) fn preserves_implicit_zero(self) -> bool {
        !matches!(self, Self::Divide)
    }
}

macro_rules! float_kernels {
    ($ty:ty, $binary:ident, $scale:ident, $avx_binary:ident, $avx_scale:ident, $wide_binary:ident, $wide_scale:ident) => {
        pub(crate) fn $binary(left: &[$ty], right: &[$ty], output: &mut [$ty], op: BinaryOp) {
            assert_eq!(left.len(), right.len());
            assert_eq!(left.len(), output.len());
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if output.len() >= 128 && std::is_x86_feature_detected!("avx512f") {
                // SAFETY: these floating kernels require AVX-512F only, checked
                // above; validated slice lengths cover every load/store.
                unsafe { x86::$wide_binary(left, right, output, op) };
                return;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if output.len() >= 32 && std::is_x86_feature_detected!("avx2") {
                // SAFETY: feature detection guards the call; equal, disjoint
                // borrowed slices establish the kernel's memory preconditions.
                unsafe { x86::$avx_binary(left, right, output, op) };
                return;
            }
            for ((out, &a), &b) in output.iter_mut().zip(left).zip(right) {
                *out = op.apply(a, b);
            }
        }

        pub(crate) fn $scale(values: &mut [$ty], scalar: $ty) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if values.len() >= 128 && std::is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F is detected; all accesses are within values.
                unsafe { x86::$wide_scale(values, scalar) };
                return;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if values.len() >= 32 && std::is_x86_feature_detected!("avx2") {
                // SAFETY: the feature is available and the borrowed slice is
                // valid for every full-width load/store and scalar tail.
                unsafe { x86::$avx_scale(values, scalar) };
                return;
            }
            for value in values {
                *value *= scalar;
            }
        }
    };
}

float_kernels!(
    f32,
    binary_f32,
    scale_f32,
    binary_f32,
    scale_f32,
    binary_f32_512,
    scale_f32_512
);
float_kernels!(
    f64,
    binary_f64,
    scale_f64,
    binary_f64,
    scale_f64,
    binary_f64_512,
    scale_f64_512
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use super::BinaryOp;

    macro_rules! packed {
        ($feature:literal, $ty:ty, $lanes:expr, $binary:ident, $scale:ident, $load:ident, $store:ident,
         $splat:ident, $add:ident, $sub:ident, $mul:ident, $div:ident) => {
            /// Requires the declared CPU feature and equally sized, valid slices.
            #[target_feature(enable = $feature)]
            pub(super) unsafe fn $binary(
                left: &[$ty],
                right: &[$ty],
                output: &mut [$ty],
                op: BinaryOp,
            ) {
                let end = output.len() / $lanes * $lanes;
                for index in (0..end).step_by($lanes) {
                    // SAFETY: each block lies completely inside the validated
                    // slices. Unaligned instructions impose no extra alignment.
                    unsafe {
                        let a = $load(left.as_ptr().add(index));
                        let b = $load(right.as_ptr().add(index));
                        let result = match op {
                            BinaryOp::Add => $add(a, b),
                            BinaryOp::Subtract => $sub(a, b),
                            BinaryOp::Multiply => $mul(a, b),
                            BinaryOp::Divide => $div(a, b),
                        };
                        $store(output.as_mut_ptr().add(index), result);
                    }
                }
                for index in end..output.len() {
                    output[index] = op.apply(left[index], right[index]);
                }
            }

            /// Requires the declared CPU feature and a valid mutable slice.
            #[target_feature(enable = $feature)]
            pub(super) unsafe fn $scale(values: &mut [$ty], scalar: $ty) {
                let end = values.len() / $lanes * $lanes;
                let factor = $splat(scalar);
                for index in (0..end).step_by($lanes) {
                    // SAFETY: the block is within the slice; unaligned access
                    // supports all ordinary Rust slice addresses.
                    unsafe {
                        let pointer = values.as_mut_ptr().add(index);
                        $store(pointer, $mul($load(pointer), factor));
                    }
                }
                for value in &mut values[end..] {
                    *value *= scalar;
                }
            }
        };
    }

    packed!(
        "avx2",
        f32,
        8,
        binary_f32,
        scale_f32,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_set1_ps,
        _mm256_add_ps,
        _mm256_sub_ps,
        _mm256_mul_ps,
        _mm256_div_ps
    );
    packed!(
        "avx2",
        f64,
        4,
        binary_f64,
        scale_f64,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_set1_pd,
        _mm256_add_pd,
        _mm256_sub_pd,
        _mm256_mul_pd,
        _mm256_div_pd
    );
    packed!(
        "avx512f",
        f32,
        16,
        binary_f32_512,
        scale_f32_512,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_set1_ps,
        _mm512_add_ps,
        _mm512_sub_ps,
        _mm512_mul_ps,
        _mm512_div_ps
    );
    packed!(
        "avx512f",
        f64,
        8,
        binary_f64_512,
        scale_f64_512,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_set1_pd,
        _mm512_add_pd,
        _mm512_sub_pd,
        _mm512_mul_pd,
        _mm512_div_pd
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! check_float {
        ($name:ident, $ty:ty, $binary:ident, $scale:ident, $wide_binary:ident, $wide_scale:ident) => {
            #[test]
            fn $name() {
                let edge = [
                    0.0,
                    -0.0,
                    1.0,
                    -2.0,
                    <$ty>::MIN_POSITIVE,
                    <$ty>::from_bits(1),
                    <$ty>::MAX,
                    <$ty>::INFINITY,
                    <$ty>::NEG_INFINITY,
                    <$ty>::NAN,
                ];
                for size in 0..=131 {
                    let left: Vec<$ty> = (0..size + 1).map(|i| edge[i % edge.len()]).collect();
                    let right: Vec<$ty> =
                        (0..size + 1).map(|i| edge[(i + 3) % edge.len()]).collect();
                    for op in [
                        BinaryOp::Add,
                        BinaryOp::Subtract,
                        BinaryOp::Multiply,
                        BinaryOp::Divide,
                    ] {
                        let mut out = vec![99.0; size + 2];
                        $binary(&left[1..], &right[1..], &mut out[1..size + 1], op);
                        for i in 0..size {
                            let expected = op.apply(left[i + 1], right[i + 1]);
                            assert!(
                                out[i + 1].to_bits() == expected.to_bits()
                                    || out[i + 1].is_nan() && expected.is_nan()
                            );
                        }
                        assert_eq!(out[0], 99.0);
                        assert_eq!(out[size + 1], 99.0);
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        if std::is_x86_feature_detected!("avx2") {
                            // SAFETY: checked CPU support, equal subslice lengths.
                            unsafe {
                                x86::$binary(&left[1..], &right[1..], &mut out[1..size + 1], op)
                            };
                            for i in 0..size {
                                let expected = op.apply(left[i + 1], right[i + 1]);
                                assert!(
                                    out[i + 1].to_bits() == expected.to_bits()
                                        || out[i + 1].is_nan() && expected.is_nan()
                                );
                            }
                        }
                    }
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if std::is_x86_feature_detected!("avx512f") {
                        for op in [
                            BinaryOp::Add,
                            BinaryOp::Subtract,
                            BinaryOp::Multiply,
                            BinaryOp::Divide,
                        ] {
                            let mut out = vec![0.0; size];
                            // SAFETY: exact required feature and matching slices.
                            unsafe { x86::$wide_binary(&left[1..], &right[1..], &mut out, op) };
                            for i in 0..size {
                                let expected = op.apply(left[i + 1], right[i + 1]);
                                assert!(
                                    out[i].to_bits() == expected.to_bits()
                                        || out[i].is_nan() && expected.is_nan()
                                );
                            }
                        }
                        let mut out = left.clone();
                        // SAFETY: feature detection and valid mutable slice.
                        unsafe { x86::$wide_scale(&mut out[1..], -1.5) };
                        for i in 1..out.len() {
                            let expected = left[i] * -1.5;
                            assert!(
                                out[i].to_bits() == expected.to_bits()
                                    || out[i].is_nan() && expected.is_nan()
                            );
                        }
                    }
                    let mut scaled = left.clone();
                    $scale(&mut scaled[1..], -1.5);
                    for (&actual, &input) in scaled[1..].iter().zip(&left[1..]) {
                        let expected = input * -1.5;
                        assert!(
                            actual.to_bits() == expected.to_bits()
                                || actual.is_nan() && expected.is_nan()
                        );
                    }
                }
            }
        };
    }

    check_float!(
        f32_edges_and_unaligned_tails,
        f32,
        binary_f32,
        scale_f32,
        binary_f32_512,
        scale_f32_512
    );
    check_float!(
        f64_edges_and_unaligned_tails,
        f64,
        binary_f64,
        scale_f64,
        binary_f64_512,
        scale_f64_512
    );
}

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

/// Allocates exactly the result buffer, then initializes it once in disjoint
/// chunks. No clone/zero pass touches pages before workers write their results.
macro_rules! owned_scale {
    ($ty:ty, $name:ident, $narrow:ident, $wide:ident) => {
        pub(crate) fn $name(input: &[$ty], scalar: $ty) -> Vec<$ty> {
            let mut output = Vec::<$ty>::with_capacity(input.len());
            let spare = &mut output.spare_capacity_mut()[..input.len()];
            crate::threading::for_each_chunk_mut_with_minimum(spare, 1, 262_144, |start, chunk| {
                let source = &input[start..start + chunk.len()];
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if chunk.len() >= 128 && std::is_x86_feature_detected!("avx512f") {
                    // SAFETY: detected feature and equal source/destination lengths.
                    unsafe { x86::$wide(source, chunk, scalar) };
                    return;
                }
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if chunk.len() >= 32 && std::is_x86_feature_detected!("avx2") {
                    // SAFETY: detected feature and equal source/destination lengths.
                    unsafe { x86::$narrow(source, chunk, scalar) };
                    return;
                }
                for (destination, &value) in chunk.iter_mut().zip(source) {
                    destination.write(value * scalar);
                }
            });
            // SAFETY: every disjoint chunk initialized all of its elements, and
            // Rayon completed before this point. On panic Vec still has length
            // zero, so no uninitialized value is ever exposed or dropped.
            unsafe { output.set_len(input.len()) };
            output
        }
    };
}
owned_scale!(f32, scaled_f32, copy_scale_f32, copy_scale_f32_512);
owned_scale!(f64, scaled_f64, copy_scale_f64, copy_scale_f64_512);

/// Flat all-active Euler update. Selection remains outside this kernel; each
/// component uses separate multiplication and addition, preserving old velocity
/// for explicit Euler. No padding, packing or extra particle buffers are needed.
pub(crate) fn euler_f64(
    position: &mut [f64],
    velocity: &mut [f64],
    acceleration: &[f64],
    dt: f64,
    semi: bool,
) {
    assert_eq!(position.len(), velocity.len());
    assert_eq!(position.len(), acceleration.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if position.len() >= 128 && std::is_x86_feature_detected!("avx512f") {
        // SAFETY: feature detection, equal lengths and disjoint mutable borrows.
        unsafe { x86::euler_f64_512(position, velocity, acceleration, dt, semi) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if position.len() >= 32 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: feature detection, equal lengths and disjoint mutable borrows.
        unsafe { x86::euler_f64_256(position, velocity, acceleration, dt, semi) };
        return;
    }
    for ((r, v), &a) in position.iter_mut().zip(velocity).zip(acceleration) {
        let old = *v;
        *v += a * dt;
        *r += if semi { *v * dt } else { old * dt };
    }
}

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
    macro_rules! copy_scale {
        ($feature:literal, $ty:ty, $lanes:expr, $name:ident, $load:ident, $store:ident, $splat:ident, $mul:ident) => {
            /// Requires the declared feature and equally sized disjoint slices.
            #[target_feature(enable = $feature)]
            pub(super) unsafe fn $name(
                input: &[$ty],
                output: &mut [std::mem::MaybeUninit<$ty>],
                scalar: $ty,
            ) {
                let end = input.len() / $lanes * $lanes;
                let factor = $splat(scalar);
                for index in (0..end).step_by($lanes) {
                    // SAFETY: complete blocks lie within both slices. Stores
                    // initialize MaybeUninit elements without reading them.
                    unsafe {
                        $store(
                            output.as_mut_ptr().add(index).cast::<$ty>(),
                            $mul($load(input.as_ptr().add(index)), factor),
                        )
                    };
                }
                for (destination, &value) in output[end..].iter_mut().zip(&input[end..]) {
                    destination.write(value * scalar);
                }
            }
        };
    }
    copy_scale!(
        "avx2",
        f32,
        8,
        copy_scale_f32,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_set1_ps,
        _mm256_mul_ps
    );
    copy_scale!(
        "avx2",
        f64,
        4,
        copy_scale_f64,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_set1_pd,
        _mm256_mul_pd
    );
    copy_scale!(
        "avx512f",
        f32,
        16,
        copy_scale_f32_512,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_set1_ps,
        _mm512_mul_ps
    );
    copy_scale!(
        "avx512f",
        f64,
        8,
        copy_scale_f64_512,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_set1_pd,
        _mm512_mul_pd
    );

    macro_rules! euler {
        ($feature:literal, $lanes:expr, $name:ident, $load:ident, $store:ident, $splat:ident, $mul:ident, $add:ident) => {
            /// Requires the declared feature and equal, disjoint valid slices.
            #[target_feature(enable = $feature)]
            pub(super) unsafe fn $name(
                position: &mut [f64],
                velocity: &mut [f64],
                acceleration: &[f64],
                dt: f64,
                semi: bool,
            ) {
                let end = position.len() / $lanes * $lanes;
                let dt_vector = $splat(dt);
                for index in (0..end).step_by($lanes) {
                    // SAFETY: full blocks lie within each validated slice; only
                    // unaligned loads/stores are used, and inputs cannot alias outputs.
                    unsafe {
                        let r = $load(position.as_ptr().add(index));
                        let old_v = $load(velocity.as_ptr().add(index));
                        let a = $load(acceleration.as_ptr().add(index));
                        let v = $add(old_v, $mul(a, dt_vector));
                        let r = $add(r, $mul(if semi { v } else { old_v }, dt_vector));
                        $store(velocity.as_mut_ptr().add(index), v);
                        $store(position.as_mut_ptr().add(index), r);
                    }
                }
                for ((r, v), &a) in position[end..]
                    .iter_mut()
                    .zip(&mut velocity[end..])
                    .zip(&acceleration[end..])
                {
                    let old = *v;
                    *v += a * dt;
                    *r += if semi { *v * dt } else { old * dt };
                }
            }
        };
    }
    euler!(
        "avx2",
        4,
        euler_f64_256,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_set1_pd,
        _mm256_mul_pd,
        _mm256_add_pd
    );
    euler!(
        "avx512f",
        8,
        euler_f64_512,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_set1_pd,
        _mm512_mul_pd,
        _mm512_add_pd
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
    #[test]
    fn owned_scale_initialization_and_euler_match_scalar_edges() {
        let edges = [
            0.0_f64,
            -0.0,
            1.0,
            -2.0,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        fn equal(a: f64, b: f64) {
            assert!(
                a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(),
                "{a:?} != {b:?}"
            );
        }
        for size in 0..=263 {
            let input: Vec<_> = (0..size + 2)
                .map(|index| edges[index % edges.len()])
                .collect();
            let source = &input[1..size + 1];
            let scaled = scaled_f64(source, -1.5);
            for (&a, &b) in scaled.iter().zip(source) {
                equal(a, b * -1.5);
            }
            let input32: Vec<_> = source.iter().map(|value| *value as f32).collect();
            for (&actual, &value) in scaled_f32(&input32, -1.5).iter().zip(&input32) {
                let expected = value * -1.5;
                assert!(
                    actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan()
                );
            }
            for semi in [false, true] {
                let mut positions = input.clone();
                let mut velocities = input.clone();
                euler_f64(
                    &mut positions[1..size + 1],
                    &mut velocities[1..size + 1],
                    source,
                    0.125,
                    semi,
                );
                equal(positions[0], input[0]);
                equal(positions[size + 1], input[size + 1]);
                equal(velocities[0], input[0]);
                equal(velocities[size + 1], input[size + 1]);
                for index in 1..size + 1 {
                    let velocity = input[index] + input[index] * 0.125;
                    let position = input[index]
                        + if semi {
                            velocity * 0.125
                        } else {
                            input[index] * 0.125
                        };
                    equal(velocities[index], velocity);
                    equal(positions[index], position);
                }
            }
        }
    }
}

//! Direct comparisons of every explicit SIMD kernel, compiled as crate unit tests.
//! See ../SIMD.md and ../run_simd.sh for reproducible scalar builds and logging.
use super::*;
use std::hint::black_box;
use std::mem::MaybeUninit;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Backend {
    Scalar,
    Avx2,
    Avx512,
    Auto,
}

impl Backend {
    fn supported(self) -> bool {
        match self {
            Self::Scalar | Self::Auto => true,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => std::is_x86_feature_detected!("avx2"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx512 => std::is_x86_feature_detected!("avx512f"),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            _ => false,
        }
    }
}

fn backends() -> Vec<Backend> {
    [
        Backend::Scalar,
        Backend::Avx2,
        Backend::Avx512,
        Backend::Auto,
    ]
    .into_iter()
    .filter(|backend| {
        if !backend.supported() {
            println!("SKIP backend={backend:?}: CPU/OS feature unavailable");
            false
        } else {
            true
        }
    })
    .collect()
}

const OPS: [(BinaryOp, &str); 4] = [
    (BinaryOp::Add, "add"),
    (BinaryOp::Subtract, "subtract"),
    (BinaryOp::Multiply, "multiply"),
    (BinaryOp::Divide, "divide"),
];

fn sizes() -> Vec<usize> {
    // Every lane remainder, both runtime thresholds, and larger odd tails.
    (0..=145)
        .chain([255, 256, 257, 263, 1023, 1024, 1025, 4099])
        .collect()
}

fn random_bits(index: usize, salt: u64) -> u64 {
    // Stateless deterministic SplitMix64; independent of RNG library versions.
    let mut value = (index as u64)
        .wrapping_add(salt)
        .wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

macro_rules! float_suite {
    ($module:ident, $ty:ty, $bits:ty, $binary:ident, $scale:ident, $owned:ident,
     $binary512:ident, $scale512:ident, $copy:ident, $copy512:ident) => {
        mod $module {
            use super::*;

            fn edges() -> [$ty; 16] {
                [0.0, -0.0, 1.0, -1.0, 1.5, -2.75, <$ty>::EPSILON,
                 <$ty>::MIN_POSITIVE, <$ty>::from_bits(1), -<$ty>::from_bits(1),
                 <$ty>::MAX, -<$ty>::MAX, <$ty>::INFINITY, <$ty>::NEG_INFINITY,
                 <$ty>::NAN, <$ty>::from_bits(<$ty>::NAN.to_bits() | 123)]
            }

            fn data(size: usize, dataset: usize, right: bool) -> Vec<$ty> {
                let edge = edges();
                (0..size).map(|i| match dataset {
                    // Together these cover all 16 x 16 ordered IEEE edge pairs.
                    0 => edge[if right { (i / edge.len()) % edge.len() } else { i % edge.len() }],
                    1 => <$ty>::from_bits(random_bits(i, if right { 71 } else { 19 }) as $bits),
                    _ => ((random_bits(i, if right { 71 } else { 19 }) % 20001) as $ty - 10000.0) / 127.0,
                }).collect()
            }

            fn equal(actual: &[$ty], expected: &[$ty], context: &str) {
                assert_eq!(actual.len(), expected.len(), "{context}: length");
                for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
                    assert!(a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()),
                        "{context}: index={i}, actual={a:?} bits={:#x}, expected={e:?} bits={:#x}",
                        a.to_bits(), e.to_bits());
                }
            }

            fn binary(backend: Backend, a: &[$ty], b: &[$ty], out: &mut [$ty], op: BinaryOp) {
                assert!(backend.supported());
                assert_eq!(a.len(), b.len());
                assert_eq!(a.len(), out.len());
                match backend {
                    Backend::Scalar => {
                        for ((o, &a), &b) in out.iter_mut().zip(a).zip(b) {
                            *o = match op {
                                BinaryOp::Add => a + b,
                                BinaryOp::Subtract => a - b,
                                BinaryOp::Multiply => a * b,
                                BinaryOp::Divide => a / b,
                            };
                        }
                    }
                    Backend::Auto => $binary(a, b, out, op),
                    // SAFETY: CPU/OS support and equal lengths checked above;
                    // Rust borrows guarantee disjoint mutable destinations.
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    Backend::Avx2 => unsafe { x86::$binary(a, b, out, op) },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    Backend::Avx512 => unsafe { x86::$binary512(a, b, out, op) },
                    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                    _ => unreachable!("unsupported backend"),
                }
            }

            fn scale(backend: Backend, values: &mut [$ty], factor: $ty) {
                assert!(backend.supported());
                match backend {
                    Backend::Scalar => values.iter_mut().for_each(|v| *v *= factor),
                    Backend::Auto => $scale(values, factor),
                    // SAFETY: CPU/OS support checked above; valid mutable slice.
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    Backend::Avx2 => unsafe { x86::$scale(values, factor) },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    Backend::Avx512 => unsafe { x86::$scale512(values, factor) },
                    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                    _ => unreachable!("unsupported backend"),
                }
            }

            fn copy_scale(backend: Backend, input: &[$ty], out: &mut [MaybeUninit<$ty>], factor: $ty) {
                assert!(backend.supported());
                assert_eq!(input.len(), out.len());
                match backend {
                    Backend::Scalar => {
                        for (o, &v) in out.iter_mut().zip(input) { o.write(v * factor); }
                    }
                    Backend::Auto => {
                        for (o, v) in out.iter_mut().zip($owned(input, factor)) { o.write(v); }
                    }
                    // SAFETY: CPU/OS support and equal slice lengths checked above.
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    Backend::Avx2 => unsafe { x86::$copy(input, out, factor) },
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    Backend::Avx512 => unsafe { x86::$copy512(input, out, factor) },
                    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                    _ => unreachable!("unsupported backend"),
                }
            }

            fn owned_scale(backend: Backend, input: &[$ty], factor: $ty) -> Vec<$ty> {
                if backend == Backend::Auto { return $owned(input, factor); }
                let mut output = Vec::<$ty>::with_capacity(input.len());
                copy_scale(backend, input, &mut output.spare_capacity_mut()[..input.len()], factor);
                // SAFETY: copy_scale initializes every element before returning.
                unsafe { output.set_len(input.len()); }
                output
            }

            #[test]
            fn arithmetic_scale_and_copy_match_all_backends() {
                let backends = backends();
                let start = Instant::now();
                let mut cases = [0_usize; 6];
                let mut elements = 0_usize;
                for size in sizes() {
                    for offset in 0..16 {
                        for dataset in 0..3 {
                            // Different input/output offsets, including every 64-byte alignment.
                            let a_storage = data(size + 32, dataset, false);
                            let b_storage = data(size + 32, dataset, true);
                            let a = &a_storage[offset..offset + size];
                            let b_offset = (offset * 3 + 1) % 16;
                            let b = &b_storage[b_offset..b_offset + size];
                            let out_offset = (offset * 7 + 3) % 16 + 1;
                            for (operation, (op, name)) in OPS.into_iter().enumerate() {
                                let mut reference = vec![0.0; size];
                                binary(Backend::Scalar, a, b, &mut reference, op);
                                for &backend in &backends {
                                    let context = format!("{} {name} {backend:?} size={size} offset={offset} dataset={dataset}", stringify!($ty));
                                    let mut out = vec![91.25; size + 33];
                                    binary(backend, a, b, &mut out[out_offset..out_offset + size], op);
                                    equal(&out[out_offset..out_offset + size], &reference, &context);
                                    assert!(out[..out_offset].iter().chain(&out[out_offset + size..]).all(|&v| v == 91.25), "{context}: guard overwritten");
                                    cases[operation] += 1;
                                    elements += size;
                                }
                            }
                            for factor in [0.0, -0.0, 1.0, -1.5, <$ty>::MIN_POSITIVE, <$ty>::MAX, <$ty>::INFINITY, <$ty>::NAN] {
                                let reference: Vec<_> = a.iter().map(|&v| v * factor).collect();
                                for &backend in &backends {
                                    let context = format!("{} scale/copy {backend:?} size={size} offset={offset} dataset={dataset} factor={factor:?}", stringify!($ty));
                                    let mut out = vec![91.25; size + 33];
                                    out[out_offset..out_offset + size].copy_from_slice(a);
                                    scale(backend, &mut out[out_offset..out_offset + size], factor);
                                    equal(&out[out_offset..out_offset + size], &reference, &context);
                                    assert!(out[..out_offset].iter().chain(&out[out_offset + size..]).all(|&v| v == 91.25), "{context}: scale guard overwritten");
                                    // Poisoned but initialized storage detects missed writes without
                                    // reading uninitialized memory if a kernel regresses.
                                    let mut copy = vec![MaybeUninit::new(91.25); size + 33];
                                    copy_scale(backend, a, &mut copy[out_offset..out_offset + size], factor);
                                    // SAFETY: entire backing buffer was initialized above.
                                    let copy: Vec<_> = copy.into_iter().map(|v| unsafe { v.assume_init() }).collect();
                                    equal(&copy[out_offset..out_offset + size], &reference, &context);
                                    assert!(copy[..out_offset].iter().chain(&copy[out_offset + size..]).all(|&v| v == 91.25), "{context}: copy guard overwritten");
                                    cases[4] += 1;
                                    cases[5] += 1;
                                    elements += 2 * size;
                                }
                            }
                        }
                    }
                }
                for (name, count) in ["add", "subtract", "multiply", "divide", "scale_in_place", "copy_scale"].into_iter().zip(cases) {
                    println!("PASS type={} op={name} backends={backends:?} cases={count} mismatches=0", stringify!($ty));
                }
                println!("SUMMARY type={} compared_elements={elements} elapsed_ms={:.1} equality=bitwise_except_nan_payload", stringify!($ty), start.elapsed().as_secs_f64() * 1000.0);
            }

            #[test]
            fn allocating_scale_matches_across_chunk_boundaries_and_pools() {
                let backends = backends();
                for threads in [1, 4] {
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                    pool.install(|| {
                        for size in [0, 1, 31, 32, 33, 127, 128, 129, 262143, 262144, 262145, 1048589] {
                            let input = data(size + 1, 2, false);
                            let input = &input[1..];
                            let reference: Vec<_> = input.iter().map(|&v| v * -1.5).collect();
                            for &backend in &backends {
                                let result = owned_scale(backend, input, -1.5);
                                equal(&result, &reference, &format!("owned {} {backend:?} threads={threads} size={size}", stringify!($ty)));
                            }
                        }
                    });
                    println!("PASS type={} op=allocating_scale backends={backends:?} pool_threads={threads} sizes=12 mismatches=0", stringify!($ty));
                }
            }

            pub(super) fn timings(backends: &[Backend]) {
                for size in [31, 128, 4099, 262145, 1048589] {
                    let a = data(size, 2, false);
                    // Nonzero normal divisors keep arithmetic timings comparable.
                    let b: Vec<_> = data(size, 2, true).iter().map(|v| v.abs() + 0.5).collect();
                    for (op, name) in OPS {
                        let mut reference = vec![0.0; size];
                        binary(Backend::Scalar, &a, &b, &mut reference, op);
                        let mut baseline = 0.0;
                        for &backend in backends {
                            let mut out = vec![0.0; size];
                            binary(backend, &a, &b, &mut out, op);
                            equal(&out, &reference, "timing preflight binary");
                            let stats = measure(size, || {
                                binary(backend, black_box(&a), black_box(&b), black_box(&mut out), black_box(op));
                                black_box(&out);
                            });
                            equal(&out, &reference, "timing postflight binary");
                            report(stringify!($ty), name, backend, size, stats, &mut baseline);
                        }
                    }
                    for name in ["scale_in_place", "copy_scale", "allocating_scale"] {
                        let mut baseline = 0.0;
                        for &backend in backends {
                            // Auto copy_scale includes allocation; only compare raw copy kernels here.
                            if name == "copy_scale" && backend == Backend::Auto { continue; }
                            let mut out = a.clone();
                            let mut copy = vec![MaybeUninit::new(0.0); size];
                            let expected: Vec<_> = a.iter().map(|&v| v * -1.0).collect();
                            scale(backend, &mut out, -1.0);
                            equal(&out, &expected, "timing preflight scale");
                            out.copy_from_slice(&a);
                            let stats = measure(size, || {
                                match name {
                                    // -1 prevents overflow/underflow during repeated in-place calls.
                                    "scale_in_place" => scale(backend, black_box(&mut out), black_box(-1.0)),
                                    "copy_scale" => copy_scale(backend, black_box(&a), black_box(&mut copy), black_box(-1.0)),
                                    _ => { black_box(owned_scale(backend, black_box(&a), black_box(-1.0))); }
                                }
                                black_box(&out);
                                black_box(&copy);
                            });
                            report(stringify!($ty), name, backend, size, stats, &mut baseline);
                        }
                    }
                }
            }
        }
    };
}

float_suite!(
    float32,
    f32,
    u32,
    binary_f32,
    scale_f32,
    scaled_f32,
    binary_f32_512,
    scale_f32_512,
    copy_scale_f32,
    copy_scale_f32_512
);
float_suite!(
    float64,
    f64,
    u64,
    binary_f64,
    scale_f64,
    scaled_f64,
    binary_f64_512,
    scale_f64_512,
    copy_scale_f64,
    copy_scale_f64_512
);

fn euler(backend: Backend, r: &mut [f64], v: &mut [f64], a: &[f64], dt: f64, semi: bool) {
    assert!(backend.supported());
    assert_eq!(r.len(), v.len());
    assert_eq!(r.len(), a.len());
    match backend {
        Backend::Scalar => {
            for ((r, v), &a) in r.iter_mut().zip(v).zip(a) {
                let old_v = *v;
                let new_v = old_v + a * dt;
                *r += if semi { new_v } else { old_v } * dt;
                *v = new_v;
            }
        }
        Backend::Auto => euler_f64(r, v, a, dt, semi),
        // SAFETY: checked CPU/OS support, equal lengths and disjoint mutable borrows.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx2 => unsafe { x86::euler_f64_256(r, v, a, dt, semi) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx512 => unsafe { x86::euler_f64_512(r, v, a, dt, semi) },
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        _ => unreachable!("unsupported backend"),
    }
}

#[test]
fn euler_variants_match_all_backends_and_repeated_steps() {
    let backends = backends();
    let mut cases = [0_usize; 2];
    let edge = [
        0.0,
        -0.0,
        1.0,
        -2.75,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    for size in sizes().into_iter().chain([16383, 16384, 16385]) {
        for offset in 0..8 {
            for dataset in 0..3 {
                let data = |salt: usize| -> Vec<f64> {
                    (0..size + 17)
                        .map(|i| match dataset {
                            0 => edge[(i + salt) % edge.len()],
                            1 => f64::from_bits(random_bits(i, salt as u64)),
                            _ => ((random_bits(i, salt as u64) % 20001) as f64 - 10000.0) / 127.0,
                        })
                        .collect()
                };
                let positions = data(3);
                let velocities = data(7);
                let acceleration = data(11);
                let span = offset + 1..offset + 1 + size;
                for semi in [false, true] {
                    for dt in [
                        0.0,
                        -0.0,
                        0.125,
                        -0.3,
                        f64::MIN_POSITIVE,
                        f64::MAX,
                        f64::INFINITY,
                        f64::NAN,
                    ] {
                        let mut expected_r = positions.clone();
                        let mut expected_v = velocities.clone();
                        let steps = if dataset == 2 && dt == 0.125 { 17 } else { 1 };
                        for _ in 0..steps {
                            euler(
                                Backend::Scalar,
                                &mut expected_r[span.clone()],
                                &mut expected_v[span.clone()],
                                &acceleration[span.clone()],
                                dt,
                                semi,
                            );
                        }
                        for &backend in &backends {
                            let mut r = positions.clone();
                            let mut v = velocities.clone();
                            for _ in 0..steps {
                                euler(
                                    backend,
                                    &mut r[span.clone()],
                                    &mut v[span.clone()],
                                    &acceleration[span.clone()],
                                    dt,
                                    semi,
                                );
                            }
                            for (field, actual, expected) in
                                [("position", &r, &expected_r), ("velocity", &v, &expected_v)]
                            {
                                for (i, (&actual, &expected)) in
                                    actual.iter().zip(expected).enumerate()
                                {
                                    assert!(
                                        actual.to_bits() == expected.to_bits()
                                            || actual.is_nan() && expected.is_nan(),
                                        "Euler {backend:?} semi={semi} size={size} offset={offset} dataset={dataset} dt={dt:?} steps={steps} {field}[{i}] actual={actual:?} expected={expected:?}"
                                    );
                                    if !span.contains(&i) {
                                        assert_eq!(
                                            actual.to_bits(),
                                            expected.to_bits(),
                                            "Euler guard modified"
                                        );
                                    }
                                }
                            }
                            cases[usize::from(semi)] += 1;
                        }
                    }
                }
            }
        }
    }
    for (name, count) in ["explicit_euler", "semi_implicit_euler"]
        .into_iter()
        .zip(cases)
    {
        println!(
            "PASS type=f64 op={name} backends={backends:?} cases={count} steps=1,17 mismatches=0"
        );
    }
}

// Timings deliberately have no speed assertions: CPU frequency, load, and AVX-512
// throttling vary. Allocation is included only in the allocating_scale rows.
fn measure(size: usize, mut operation: impl FnMut()) -> [f64; 3] {
    let iterations = (262144 / size.max(1)).clamp(2, 1000);
    for _ in 0..3 {
        operation();
    }
    let mut samples = [0.0; 7];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        *sample = start.elapsed().as_secs_f64() * 1e9 / iterations as f64;
    }
    samples.sort_by(f64::total_cmp);
    [samples[3], samples[0], samples[6]]
}

fn report(ty: &str, op: &str, backend: Backend, size: usize, stats: [f64; 3], baseline: &mut f64) {
    if backend == Backend::Scalar {
        *baseline = stats[0];
    }
    println!(
        "TIMING,{ty},{op},{backend:?},{size},{:.2},{:.2},{:.2},{:.3},{:.3}",
        stats[0],
        stats[1],
        stats[2],
        stats[0] / size as f64,
        *baseline / stats[0]
    );
}

#[test]
#[ignore = "opt-in release timings: bash tests/run_simd.sh --timings"]
fn backend_timings() {
    assert!(!cfg!(debug_assertions), "Run timings in release mode");
    let backends = backends();
    println!(
        "TIMING,type,operation,backend,elements,median_ns,min_ns,max_ns,ns_per_element,speedup_vs_scalar"
    );
    println!(
        "TIMING_INFO samples=7 warmups=3 pool_threads=1; use tests/run_simd.sh to disable scalar auto-vectorization"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        float32::timings(&backends);
        float64::timings(&backends);
        for size in [31, 128, 4099, 262145, 1048589] {
            for semi in [false, true] {
                let mut baseline = 0.0;
                for &backend in &backends {
                    let mut r = vec![1.25; size];
                    let mut v = vec![0.5; size];
                    let a = vec![0.125; size];
                    let stats = measure(size, || {
                        euler(
                            backend,
                            black_box(&mut r),
                            black_box(&mut v),
                            black_box(&a),
                            black_box(0.001),
                            semi,
                        );
                        black_box(&r);
                        black_box(&v);
                    });
                    report(
                        "f64",
                        if semi {
                            "semi_implicit_euler"
                        } else {
                            "explicit_euler"
                        },
                        backend,
                        size,
                        stats,
                        &mut baseline,
                    );
                }
            }
        }
    });
}

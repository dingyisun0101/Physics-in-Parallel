//! Dependency-free, opt-in release measurements. Run with `cargo bench --bench
//! kernels -- --run`; all-target tests compile this without timing workloads.
use physics_in_parallel::prelude::basic::*;
use physics_in_parallel::prelude::models::*;
use std::hint::black_box;
use std::time::Instant;

fn measure(name: &str, iterations: usize, mut operation: impl FnMut()) {
    for _ in 0..3 {
        operation();
    }
    let mut samples = [0.0; 5];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        *sample = start.elapsed().as_secs_f64() * 1e9 / iterations as f64;
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{name},{:.1},{:.1},{:.1}",
        samples[2], samples[0], samples[4]
    );
}

fn containers() {
    for size in [3, 127, 100_003, 1_000_003] {
        let a = Tensor::filled(&[size], Backend::Dense, 1.25_f64).unwrap();
        let b = Tensor::filled(&[size], Backend::Dense, 2.5_f64).unwrap();
        let mut out = a.clone();
        let iterations = if size < 1000 {
            3000
        } else if size < 1_000_000 {
            100
        } else {
            20
        };
        measure(&format!("f64_add_into_{size}"), iterations, || {
            black_box(&a)
                .add_into(black_box(&b), black_box(&mut out))
                .unwrap();
            black_box(out.get(&[size - 1]).unwrap());
        });
        assert_eq!(out.get(&[size - 1]).unwrap(), 3.75);
        measure(&format!("f64_scale_allocating_{size}"), iterations, || {
            black_box(black_box(&a).scale(black_box(1.5)));
        });
        let a32 = a.cast::<f32>().unwrap();
        let b32 = b.cast::<f32>().unwrap();
        let mut out32 = a32.clone();
        measure(&format!("f32_add_into_{size}"), iterations, || {
            a32.add_into(black_box(&b32), black_box(&mut out32))
                .unwrap();
            black_box(out32.get(&[size - 1]).unwrap());
        });
    }
    let sparse =
        Tensor::from_entries(&[1_000_000_000], Backend::Sparse, vec![(vec![42], 2.0_f64)]).unwrap();
    measure("sparse_scale_1_of_1e9", 3000, || {
        black_box(sparse.scale(black_box(2.0)));
    });
    for (m, k, n, iterations) in [(4, 4, 4, 1000), (64, 96, 48, 20), (128, 128, 128, 5)] {
        let left: Vec<_> = (0..m * k).map(|i| (i % 13) as f64 * 0.1).collect();
        let right: Vec<_> = (0..k * n).map(|i| (i % 17) as f64 * 0.1).collect();
        let a = Matrix::from_values(m, k, Backend::Dense, left.clone()).unwrap();
        let b = Matrix::from_values(k, n, Backend::Dense, right.clone()).unwrap();
        let mut result = Matrix::zeros(m, n, Backend::Dense).unwrap();
        let mut reference = vec![0.0; m * n];
        let reference_op = |out: &mut [f64]| {
            for row in 0..m {
                for column in 0..n {
                    out[row * n + column] = (0..k).fold(0.0, |sum, inner| {
                        sum + left[row * k + inner] * right[inner * n + column]
                    });
                }
            }
        };
        measure(&format!("matmul_into_{m}x{k}x{n}"), iterations, || {
            a.matmul_into(black_box(&b), black_box(&mut result))
                .unwrap();
            black_box(&result);
        });
        measure(
            &format!("matmul_slice_reference_{m}x{k}x{n}"),
            iterations,
            || {
                reference_op(black_box(&mut reference));
                black_box(&reference);
            },
        );
        assert_eq!(result.values().collect::<Vec<_>>(), reference);
        measure(&format!("transpose_{m}x{k}"), iterations, || {
            black_box(a.transpose().unwrap());
        });
    }
}

fn spatial_and_models() {
    let geometry =
        SquareLatticeGeometry::new(&[128, 128], BoundaryCondition::Periodic, None).unwrap();
    let input: Vec<_> = (0..128 * 128).map(|i| (i as f64).sin()).collect();
    let mut output = vec![0.0; input.len()];
    measure("laplacian_128x128", 100, || {
        geometry
            .laplacian(black_box(&input), 1, black_box(&mut output))
            .unwrap();
    });
    let rng = ResolvedRng::new(17, RngMethod::IndexedSplitMix64);
    let mut pairs =
        PairGenerator::new(&[128, 128], PairingMethod::IndependentUniform, 10_000, rng).unwrap();
    measure("pair_refresh_10000", 30, || {
        pairs.refresh_at(black_box(9));
        black_box(pairs.targets());
    });
    let mut particles = create_template(3, 10_000).unwrap();
    particles.attribute_mut::<f64>(ATTR_V).unwrap().fill(0.25);
    measure("explicit_euler_10000x3", 100, || {
        ExplicitEuler
            .apply(black_box(&mut particles), 0.001)
            .unwrap();
    });
    measure("kinetic_summary_10000x3", 100, || {
        black_box(kinetic_summary(black_box(&particles), ParticleSelection::AliveOnly).unwrap());
    });
    let mut springs = SpringNetwork::new();
    let law = Spring::new(1.0, 0.5, None).unwrap();
    for i in 0..10_000 {
        particles
            .set_attribute_vector(ATTR_R, i, &[i as f64 * 0.01, 0.0, 0.0])
            .unwrap();
    }
    springs
        .insert_many((0..9999).map(|i| ((i, i + 1), law)))
        .unwrap();
    measure("spring_force_9999_edges", 50, || {
        particles.attribute_mut::<f64>(ATTR_A).unwrap().fill(0.0);
        springs
            .apply(black_box(&mut particles), ParticleSelection::AliveOnly)
            .unwrap();
    });
    measure("graph_replace_batch_4_of_9999", 1000, || {
        springs
            .insert_many((0..4).map(|i| ((i, i + 1), law)))
            .unwrap();
    });
    let mut neighbors = ParticleNeighborList::from_box(&[1000.0; 3], 0.011).unwrap();
    let mut output_pairs = Vec::new();
    measure("neighbors_rebuild_query_10000", 5, || {
        neighbors
            .rebuild_and_collect_into(&particles, ParticleSelection::AliveOnly, &mut output_pairs)
            .unwrap();
    });
    assert_eq!(output_pairs.len(), 9999);
}

fn large_workloads() {
    for size in [1_000_003, 10_000_003] {
        let a = Tensor::filled(&[size], Backend::Dense, 1.25_f64).unwrap();
        let b = Tensor::filled(&[size], Backend::Dense, 2.5_f64).unwrap();
        let mut out = a.clone();
        measure(&format!("large_f64_add_into_{size}"), 3, || {
            a.add_into(black_box(&b), black_box(&mut out)).unwrap();
        });
        assert_eq!(out.get(&[size - 1]).unwrap(), 3.75);
        measure(&format!("large_f64_scale_allocating_{size}"), 3, || {
            black_box(a.scale(black_box(1.5)));
        });
    }
    let size = 512;
    let a = Matrix::filled(size, size, Backend::Dense, 0.25_f64).unwrap();
    let b = Matrix::filled(size, size, Backend::Dense, 0.5_f64).unwrap();
    let mut product = Matrix::zeros(size, size, Backend::Dense).unwrap();
    measure("large_matmul_512", 1, || {
        a.matmul_into(black_box(&b), black_box(&mut product))
            .unwrap();
    });
    assert!(product.values().all(|value| value == 64.0));
    let lattice = SquareLatticeGeometry::periodic(&[2048, 2048]).unwrap();
    let input = vec![2.0; lattice.num_sites()];
    let mut output = vec![0.0; input.len()];
    measure("large_laplacian_2048x2048", 3, || {
        lattice
            .laplacian(black_box(&input), 1, black_box(&mut output))
            .unwrap();
    });
    assert!(output.iter().all(|value| *value == 0.0));
    drop(input);
    drop(output);
    let mut particles = create_template(3, 1_000_000).unwrap();
    particles.attribute_mut::<f64>(ATTR_V).unwrap().fill(0.25);
    particles.attribute_mut::<f64>(ATTR_A).unwrap().fill(0.5);
    measure("large_euler_1000000x3", 5, || {
        ExplicitEuler
            .apply(black_box(&mut particles), 0.001)
            .unwrap();
    });
    measure("large_kinetic_1000000x3", 5, || {
        black_box(kinetic_summary(black_box(&particles), ParticleSelection::AliveOnly).unwrap());
    });
    drop(particles);
    let mut particles = create_template(3, 250_000).unwrap();
    for i in 0..250_000 {
        particles
            .set_attribute_vector(ATTR_R, i, &[i as f64 * 0.01, 0.0, 0.0])
            .unwrap();
    }
    let mut neighbors = ParticleNeighborList::from_box(&[3000.0; 3], 0.011).unwrap();
    let mut pairs = Vec::new();
    measure("large_neighbors_250000", 1, || {
        neighbors
            .rebuild_and_collect_into(&particles, ParticleSelection::AliveOnly, &mut pairs)
            .unwrap();
    });
    assert_eq!(pairs.len(), 249_999);
}

fn main() {
    if !std::env::args().any(|argument| argument == "--run") {
        println!("Pass --run to execute timing workloads.");
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    eprintln!(
        "avx2={}, avx512f={}",
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("avx512f")
    );
    println!("case,median_ns,min_ns,max_ns");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for cap in [1, 4] {
        set_max_threads(Some(cap)).unwrap();
        println!("# caller_pool=4,pip_cap={cap}");
        pool.install(|| {
            if std::env::args().any(|argument| argument == "--large") {
                large_workloads();
            } else {
                containers();
                spatial_and_models();
            }
        });
    }
    set_max_threads(None).unwrap();
}

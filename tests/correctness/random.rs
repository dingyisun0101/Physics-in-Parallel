use super::support::*;
use physics_in_parallel::prelude::advanced::*;
use physics_in_parallel::prelude::basic::*;

// Straight-line scalar reference for the versioned indexed coordinate format.
// No prepared prefixes, batched helpers, or PiP RNG calls are used by the oracle.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn word(seed: u64, step: u64, domain: u64, item: u64, component: u64, draw: u64) -> u64 {
    let mut x = mix(seed ^ 0x6a09e667f3bcc909);
    for coordinate in [step, domain, item, component, draw] {
        x = mix(x ^ mix(coordinate.wrapping_add(0x9e3779b97f4a7c15)));
    }
    x
}
pub(super) fn unit(seed: u64, step: u64, domain: u64, item: u64, component: u64, draw: u64) -> f64 {
    (word(seed, step, domain, item, component, draw) >> 11) as f64 / 9007199254740992.0
}
pub(super) fn normal(seed: u64, step: u64, domain: u64, item: u64, component: u64) -> f64 {
    let u = (word(seed, step, domain, item, component, 0) >> 11) as f64;
    let radius = (-2.0 * ((u + 0.5) / 9007199254740992.0).ln()).sqrt();
    radius * (2.0 * std::f64::consts::PI * unit(seed, step, domain, item, component, 1)).cos()
}
fn index(seed: u64, step: u64, domain: u64, item: u64, component: u64, n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let n = n as u64;
    let threshold = n.wrapping_neg() % n;
    for draw in 0..u64::MAX {
        let product = word(seed, step, domain, item, component, draw) as u128 * n as u128;
        if product as u64 >= threshold {
            return Some((product >> 64) as usize);
        }
    }
    unreachable!()
}

#[test]
fn indexed_rng_and_distributions_match_scalar_coordinate_reference() {
    let mut report = Report::default();
    assert_eq!(word(17, 2, 3, 4, 5, 6), 0x995beef154ed1885);
    for seed in [0, 17, u64::MAX] {
        let resolved = ResolvedRng::new(seed, RngMethod::IndexedSplitMix64);
        let rng = IndexedRng::new(resolved).unwrap();
        for step in [0, 1, u64::MAX] {
            for i in 0..257_u64 {
                report.close(
                    "IndexedRng.unit/normal",
                    &[
                        rng.unit_f64(step, 71, i, i % 7, 3),
                        rng.standard_normal(step, 71, i, i % 7),
                    ],
                    &[
                        unit(seed, step, 71, i, i % 7, 3),
                        normal(seed, step, 71, i, i % 7),
                    ],
                    ATOL,
                    RTOL,
                );
                for n in [0, 1, 3, 127, usize::MAX / 2 + 2, usize::MAX] {
                    report.exact(
                        "IndexedRng.uniform_index/rejection",
                        rng.uniform_index(step, 71, i, 2, n),
                        index(seed, step, 71, i, 2, n),
                    );
                }
            }
        }
        for backend in BACKENDS {
            for n in [1, 31, 32, 127, 128, 257, 16403] {
                for kind in [
                    RandType::Uniform {
                        low: -2.5,
                        high: 3.75,
                    },
                    RandType::Normal {
                        mean: 0.75,
                        std: 1.25,
                    },
                    RandType::Normal {
                        mean: -1.25,
                        std: 0.0,
                    },
                    RandType::Bernoulli { p: 0.0 },
                    RandType::Bernoulli { p: 0.37 },
                    RandType::Bernoulli { p: 1.0 },
                ] {
                    let filler = TensorRandFiller::new(kind, resolved).unwrap();
                    let mut out = Tensor::<f64>::zeros(&[n], backend).unwrap();
                    filler.fill_at(&mut out, 13, 71).unwrap();
                    let expected: Vec<_> = (0..n)
                        .map(|i| match kind {
                            RandType::Uniform { low, high } => {
                                low + (high - low) * unit(seed, 13, 71, i as u64, 0, 0)
                            }
                            RandType::Normal { mean, std } => {
                                mean + std * normal(seed, 13, 71, i as u64, 0)
                            }
                            RandType::Bernoulli { p } => {
                                f64::from(unit(seed, 13, 71, i as u64, 0, 0) < p)
                            }
                            _ => unreachable!(),
                        })
                        .collect();
                    report.close(
                        "TensorRandFiller.fill_at",
                        &out.values().collect::<Vec<_>>(),
                        &expected,
                        ATOL,
                        RTOL,
                    );
                }
            }
        }
        report.exact(
            "ResolvedRng.provenance",
            resolved.encode_seed(),
            seed.to_string(),
        );
        let decoded: IndexedRng =
            serde_json::from_str(&serde_json::to_string(&rng).unwrap()).unwrap();
        report.exact("IndexedRng.serde", decoded.resolved_rng(), resolved);
    }
    report.finish();
}

#[test]
fn stateful_fillers_match_serial_rand_streams() {
    use rand::SeedableRng;
    use rand_distr::{Bernoulli, Distribution, Normal, Uniform};
    let mut report = Report::default();
    macro_rules! method {
        ($method:ident,$rng:ty) => {{
            for n in [1, 31, 32, 33, 257, 16403] {
                for backend in BACKENDS {
                    let mut master = rand::rngs::SmallRng::seed_from_u64(71);
                    let mut lanes: Vec<$rng> =
                        (0..32).map(|_| <$rng>::from_rng(&mut master)).collect();
                    let mut filler = TensorRandFiller::new(
                        RandType::Uniform {
                            low: -2.0,
                            high: 3.0,
                        },
                        ResolvedRng::new(71, RngMethod::$method),
                    )
                    .unwrap();
                    let mut out = Tensor::<f64>::zeros(&[n], backend).unwrap();
                    for kind in [
                        RandType::Uniform {
                            low: -2.0,
                            high: 3.0,
                        },
                        RandType::Normal {
                            mean: 0.75,
                            std: 1.25,
                        },
                        RandType::Bernoulli { p: 0.37 },
                    ] {
                        filler.set_kind(kind);
                        filler.fill(&mut out).unwrap();
                        let mut expected = vec![0.0; n];
                        for (chunk, rng) in
                            expected.chunks_mut(n.div_ceil(n.min(32))).zip(&mut lanes)
                        {
                            for value in chunk {
                                *value = match kind {
                                    RandType::Uniform { low, high } => {
                                        Uniform::new(low, high).unwrap().sample(rng)
                                    }
                                    RandType::Normal { mean, std } => {
                                        Normal::new(mean, std).unwrap().sample(rng)
                                    }
                                    RandType::Bernoulli { p } => {
                                        f64::from(Bernoulli::new(p).unwrap().sample(rng))
                                    }
                                    _ => unreachable!(),
                                };
                            }
                        }
                        report.close(
                            concat!("TensorRandFiller.stateful.", stringify!($method)),
                            &out.values().collect::<Vec<_>>(),
                            &expected,
                            0.0,
                            0.0,
                        );
                    }
                    // A fresh stream checks inclusive integer bounds independently.
                    let mut master = rand::rngs::SmallRng::seed_from_u64(71);
                    let mut lanes: Vec<$rng> =
                        (0..32).map(|_| <$rng>::from_rng(&mut master)).collect();
                    let mut expected = vec![0_i64; n];
                    for (chunk, rng) in expected.chunks_mut(n.div_ceil(n.min(32))).zip(&mut lanes) {
                        for value in chunk {
                            *value = Uniform::new_inclusive(-7_i64, 11).unwrap().sample(rng);
                        }
                    }
                    let mut filler = TensorRandFiller::new(
                        RandType::UniformInt { low: -7, high: 11 },
                        ResolvedRng::new(71, RngMethod::$method),
                    )
                    .unwrap();
                    let mut out = Tensor::<i64>::zeros(&[n], backend).unwrap();
                    filler.fill(&mut out).unwrap();
                    let mut signed = Tensor::<isize>::zeros(&[n], backend).unwrap();
                    TensorRandFiller::new(
                        RandType::UniformInt { low: -7, high: 11 },
                        ResolvedRng::new(71, RngMethod::$method),
                    )
                    .unwrap()
                    .fill(&mut signed)
                    .unwrap();
                    report.exact(
                        "TensorRandFiller.isize",
                        signed.values().collect::<Vec<_>>(),
                        expected.iter().map(|&v| v as isize).collect::<Vec<_>>(),
                    );
                    report.exact(
                        concat!("TensorRandFiller.integers.", stringify!($method)),
                        out.values().collect::<Vec<_>>(),
                        expected,
                    );
                    filler.set_kind(RandType::Bernoulli { p: 0.37 });
                    filler.fill(&mut out).unwrap();
                    let mut expected = vec![0_i64; n];
                    for (chunk, rng) in expected.chunks_mut(n.div_ceil(n.min(32))).zip(&mut lanes) {
                        for value in chunk {
                            *value = i64::from(Bernoulli::new(0.37).unwrap().sample(rng));
                        }
                    }
                    report.exact(
                        "TensorRandFiller.i64 Bernoulli",
                        out.values().collect::<Vec<_>>(),
                        expected,
                    );
                    let mut master = rand::rngs::SmallRng::seed_from_u64(71);
                    let mut lanes: Vec<$rng> =
                        (0..32).map(|_| <$rng>::from_rng(&mut master)).collect();
                    let mut expected = vec![0_usize; n];
                    for (chunk, rng) in expected.chunks_mut(n.div_ceil(n.min(32))).zip(&mut lanes) {
                        for value in chunk {
                            *value = Uniform::new_inclusive(0_usize, 11).unwrap().sample(rng);
                        }
                    }
                    let mut unsigned = Tensor::<usize>::zeros(&[n], backend).unwrap();
                    TensorRandFiller::new(
                        RandType::UniformInt { low: 0, high: 11 },
                        ResolvedRng::new(71, RngMethod::$method),
                    )
                    .unwrap()
                    .fill(&mut unsigned)
                    .unwrap();
                    report.exact(
                        "TensorRandFiller.usize",
                        unsigned.values().collect::<Vec<_>>(),
                        expected,
                    );
                }
            }
        }};
    }
    method!(SmallRng, rand::rngs::SmallRng);
    method!(Pcg64, rand_pcg::Pcg64);
    method!(Pcg64Mcg, rand_pcg::Pcg64Mcg);
    method!(ChaCha8, rand_chacha::ChaCha8Rng);
    method!(ChaCha12, rand_chacha::ChaCha12Rng);
    method!(ChaCha20, rand_chacha::ChaCha20Rng);
    report.finish();
}

#[test]
fn vector_samplers_and_particle_initialization_match_formulas() {
    use physics_in_parallel::prelude::models::*;
    let mut report = Report::default();
    let seed = 19;
    let rng = ResolvedRng::new(seed, RngMethod::IndexedSplitMix64);
    for backend in BACKENDS {
        for dim in [1, 2, 3, 7] {
            for n in [1, 37, 2351] {
                let widths: Vec<_> = (0..dim).map(|i| i as f64 + 1.0).collect();
                let means = data(dim, 3);
                let stds: Vec<_> = (0..dim).map(|i| 0.125 * i as f64).collect();
                for method in [
                    VectorSamplingMethod::Uniform {
                        low: -1.5,
                        high: 2.5,
                    },
                    VectorSamplingMethod::UniformCentered { box_size: &widths },
                    VectorSamplingMethod::GaussianPerAxis {
                        mean: &means,
                        std: &stds,
                    },
                    VectorSamplingMethod::JitteredLattice {
                        spacings: &widths,
                        sigmas: &stds,
                    },
                ] {
                    let mut side = 1_usize;
                    while side.pow(dim as u32) < n {
                        side += 1;
                    }
                    let expected: Vec<_> = (0..n)
                        .flat_map(|i| {
                            (0..dim).map({
                                let method = method.clone();
                                move |c| match method {
                                    VectorSamplingMethod::Uniform { low, high } => {
                                        low + (high - low)
                                            * unit(
                                                seed,
                                                0,
                                                0x6f428f3198f7067a,
                                                i as u64,
                                                c as u64,
                                                0,
                                            )
                                    }
                                    VectorSamplingMethod::UniformCentered { box_size } => {
                                        box_size[c]
                                            * (unit(
                                                seed,
                                                0,
                                                0xb4aca77ce8dfb5f1,
                                                i as u64,
                                                c as u64,
                                                0,
                                            ) - 0.5)
                                    }
                                    VectorSamplingMethod::GaussianPerAxis { mean, std } => {
                                        mean[c]
                                            + std[c]
                                                * normal(
                                                    seed,
                                                    0,
                                                    0x15eb0cc7aac57548,
                                                    i as u64,
                                                    c as u64,
                                                )
                                    }
                                    VectorSamplingMethod::JitteredLattice { spacings, sigmas } => {
                                        (i / side.pow(c as u32) % side) as f64 * spacings[c]
                                            + sigmas[c]
                                                * normal(
                                                    seed,
                                                    0,
                                                    0xcbefa4d0a1940f13,
                                                    i as u64,
                                                    c as u64,
                                                )
                                    }
                                }
                            })
                        })
                        .collect();
                    let mut out = VectorList::zeros(dim, n, backend).unwrap();
                    sample_vectors(&mut out, method.clone(), rng).unwrap();
                    report.close(
                        "sample_vectors all methods",
                        &out.values().collect::<Vec<_>>(),
                        &expected,
                        ATOL,
                        RTOL,
                    );
                    let mut particles = create_template(dim, n).unwrap();
                    particles
                        .attribute_mut::<f64>(ATTR_R)
                        .unwrap()
                        .set_backend(backend);
                    randomize_r(&mut particles, method, rng).unwrap();
                    report.close(
                        "randomize_r all methods",
                        &particles
                            .attribute::<f64>(ATTR_R)
                            .unwrap()
                            .values()
                            .collect::<Vec<_>>(),
                        &expected,
                        ATOL,
                        RTOL,
                    );
                }
                let mut particles = create_template(dim, n).unwrap();
                particles
                    .attribute_mut::<f64>(ATTR_V)
                    .unwrap()
                    .set_backend(backend);
                for i in 0..n {
                    set_mass(&mut particles, i, 0.5 + i as f64 % 7.0).unwrap();
                }
                for method in [
                    VelocitySamplingMethod::Uniform {
                        low: -1.5,
                        high: 2.5,
                    },
                    VelocitySamplingMethod::GaussianPerAxis {
                        mean: &means,
                        std: &stds,
                    },
                    VelocitySamplingMethod::MaxwellBoltzmann { tau: 1.75 },
                    VelocitySamplingMethod::MaxwellBoltzmann { tau: 0.0 },
                ] {
                    let expected: Vec<_> = (0..n)
                        .flat_map(|i| {
                            (0..dim).map({
                                let method = method.clone();
                                move |c| match method {
                                    VelocitySamplingMethod::Uniform { low, high } => {
                                        low + (high - low)
                                            * unit(
                                                seed,
                                                0,
                                                0x6f428f3198f7067a,
                                                i as u64,
                                                c as u64,
                                                0,
                                            )
                                    }
                                    VelocitySamplingMethod::GaussianPerAxis { mean, std } => {
                                        mean[c]
                                            + std[c]
                                                * normal(
                                                    seed,
                                                    0,
                                                    0x15eb0cc7aac57548,
                                                    i as u64,
                                                    c as u64,
                                                )
                                    }
                                    VelocitySamplingMethod::MaxwellBoltzmann { tau } => {
                                        (tau / (0.5 + i as f64 % 7.0)).sqrt()
                                            * normal(
                                                seed,
                                                0,
                                                0xf4d4b3019eab6342,
                                                i as u64,
                                                c as u64,
                                            )
                                    }
                                }
                            })
                        })
                        .collect();
                    randomize_v(&mut particles, method, rng).unwrap();
                    report.close(
                        "randomize_v all methods",
                        &particles
                            .attribute::<f64>(ATTR_V)
                            .unwrap()
                            .values()
                            .collect::<Vec<_>>(),
                        &expected,
                        ATOL,
                        RTOL,
                    );
                }
            }
        }
    }
    report.finish();
}

#[test]
fn kernels_and_pair_generation_match_scalar_sampling_and_coordinates() {
    let mut report = Report::default();
    let seed = 71;
    let rng = ResolvedRng::new(seed, RngMethod::IndexedSplitMix64);
    let domain = 0xbeb394874b9ca7f5;
    for dim in [1, 2, 3, 5] {
        let shape = vec![7; dim];
        let total = shape.iter().product();
        let kernels: Vec<Box<dyn Kernel>> = vec![
            Box::new(NearestNeighborKernel::try_new(dim).unwrap()),
            Box::new(UniformDistanceKernel::try_new(4.0, 1.0).unwrap()),
            Box::new(PowerLawKernel::try_new(4.0, 1.0, 1.5).unwrap()),
        ];
        for step in [0, 7, 13] {
            for kernel in &kernels {
                let expected: Vec<_> = (0..257)
                    .map(|i| match kernel.kind() {
                        KernelType::NearestNeighbor { .. } => {
                            index(seed, step, domain, i, 0, 2 * dim).unwrap() as f64
                        }
                        KernelType::UniformDistance { l, c } => {
                            c + (l - c) * unit(seed, step, domain, i, 0, 0)
                        }
                        KernelType::PowerLaw { l, c, mu } => (c.powf(-mu)
                            + unit(seed, step, domain, i, 0, 0) * (l.powf(-mu) - c.powf(-mu)))
                        .powf(-1.0 / mu),
                    })
                    .collect();
                report.close(
                    "Kernel.batch/scalar",
                    &kernel.sample_batch_indexed(rng, step, 257).unwrap(),
                    &expected,
                    ATOL,
                    RTOL,
                );
                for i in [0, 13, 256] {
                    report.close(
                        "Kernel.sample_indexed",
                        &[kernel.sample_indexed(rng, step, i).unwrap()],
                        &[expected[i as usize]],
                        ATOL,
                        RTOL,
                    );
                }
            }
            let methods = std::iter::once(PairingMethod::IndependentUniform).chain(
                kernels.iter().flat_map(|k| {
                    [SourceMode::Origin, SourceMode::RandomUniform].map(|sources| {
                        PairingMethod::Kernel {
                            kernel: k.kind(),
                            sources,
                        }
                    })
                }),
            );
            for method in methods {
                let n = 257;
                let mut pairs = PairGenerator::new(&shape, method, n, rng).unwrap();
                pairs.refresh_at(step);
                let mut sources = Vec::new();
                let mut targets = Vec::new();
                let mut displacements = Vec::new();
                for i in 0..n {
                    let coordinate = |site| {
                        coords(site, &shape)
                            .into_iter()
                            .map(|x| x as isize)
                            .collect::<Vec<_>>()
                    };
                    let (source, delta) = match method {
                        PairingMethod::IndependentUniform => {
                            let s = coordinate(
                                index(seed, step, 0x5fc7f80b2257ae21, i as u64, 0, total).unwrap(),
                            );
                            let t = coordinate(
                                index(seed, step, 0xe895319d634cc4d3, i as u64, 0, total).unwrap(),
                            );
                            let d = t.iter().zip(&s).map(|(t, s)| t - s).collect();
                            (s, d)
                        }
                        PairingMethod::Kernel { kernel, sources } => {
                            let s = match sources {
                                SourceMode::Origin => vec![0; dim],
                                SourceMode::RandomUniform => coordinate(
                                    index(seed, step, 0x20dd7f455d924a31, i as u64, 0, total)
                                        .unwrap(),
                                ),
                            };
                            let hd = 0x64a4a4fe1a89827d;
                            let delta = if matches!(kernel, KernelType::NearestNeighbor { .. }) {
                                let code = index(seed, step, hd, i as u64, 0, 2 * dim).unwrap();
                                let mut d = vec![0; dim];
                                d[code / 2] = if code % 2 == 0 { 1 } else { -1 };
                                d
                            } else {
                                let u = unit(seed, step, domain, i as u64, 0, 0);
                                let length = match kernel {
                                    KernelType::UniformDistance { l, c } => c + (l - c) * u,
                                    KernelType::PowerLaw { l, c, mu } => (c.powf(-mu)
                                        + u * (l.powf(-mu) - c.powf(-mu)))
                                    .powf(-1.0 / mu),
                                    _ => unreachable!(),
                                };
                                let direction: Vec<_> = match dim {
                                    1 => {
                                        vec![if index(seed, step, hd, i as u64, 0, 2).unwrap() == 0
                                        {
                                            -1.0
                                        } else {
                                            1.0
                                        }]
                                    }
                                    2 => {
                                        let angle = std::f64::consts::TAU
                                            * unit(seed, step, hd, i as u64, 0, 0);
                                        vec![angle.cos(), angle.sin()]
                                    }
                                    3 => {
                                        let z = 2.0 * unit(seed, step, hd, i as u64, 0, 0) - 1.0;
                                        let phi = std::f64::consts::TAU
                                            * unit(seed, step, hd, i as u64, 1, 0);
                                        vec![
                                            (1.0 - z * z).sqrt() * phi.cos(),
                                            (1.0 - z * z).sqrt() * phi.sin(),
                                            z,
                                        ]
                                    }
                                    _ => {
                                        let mut d: Vec<_> = (0..dim)
                                            .map(|c| normal(seed, step, hd, i as u64, c as u64))
                                            .collect();
                                        let norm = d.iter().map(|x| x * x).sum::<f64>().sqrt();
                                        for v in &mut d {
                                            *v /= norm;
                                        }
                                        d
                                    }
                                };
                                direction.iter().map(|x| (x * length) as isize).collect()
                            };
                            (s, delta)
                        }
                    };
                    targets.extend(source.iter().zip(&delta).map(|(s, d)| s + d));
                    sources.extend(source);
                    displacements.extend(delta);
                }
                report.exact(
                    "PairGenerator.sources",
                    pairs.sources().values().collect::<Vec<_>>(),
                    sources,
                );
                report.exact(
                    "PairGenerator.displacements",
                    pairs.displacements().values().collect::<Vec<_>>(),
                    displacements,
                );
                report.exact(
                    "PairGenerator.targets",
                    pairs.targets().values().collect::<Vec<_>>(),
                    targets,
                );
            }
        }
    }
    report.finish();
}

#[test]
fn random_lattice_initialization_matches_linear_choice_and_shuffle() {
    let mut report = Report::default();
    let seed = 41;
    let rng = ResolvedRng::new(seed, RngMethod::IndexedSplitMix64);
    for n in [1, 7, 257, 16403] {
        let geometry = SquareLatticeGeometry::periodic(&[n]).unwrap();
        let choices = vec![-3_i64, 0, 7, 11];
        for weights in [None, Some(vec![0.0, 1.5, 2.25, 0.25])] {
            let expected: Vec<_> = (0..n)
                .map(|i| {
                    let selected = if let Some(weights) = &weights {
                        let mut ticket = unit(seed, 0, 0xc762ba71b5a78f31, i as u64, 0, 0)
                            * weights.iter().sum::<f64>();
                        let mut selected = weights.len() - 1;
                        for (j, &weight) in weights.iter().enumerate() {
                            if ticket < weight {
                                selected = j;
                                break;
                            }
                            ticket -= weight;
                        }
                        selected
                    } else {
                        index(seed, 0, 0xc762ba71b5a78f31, i as u64, 0, choices.len()).unwrap()
                    };
                    choices[selected]
                })
                .collect();
            let lattice = SquareLattice::new(
                geometry.clone(),
                SquareLatticeInitMethod::RandomChoices {
                    choices: choices.clone(),
                    weights,
                    rng,
                },
            )
            .unwrap();
            report.exact(
                "SquareLattice.RandomChoices",
                lattice.data(),
                expected.as_slice(),
            );
            assert_eq!(lattice.initialization_resolved_rng(), Some(rng));
        }
        let values: Vec<_> = (0..n as i64).collect();
        let mut expected = values.clone();
        let mut draw = 0;
        let mut remaining = n;
        while remaining > 1 {
            let choice = index(seed, 0, 0x899f2c5e12bdba4d, draw, 0, remaining).unwrap();
            let temporary = expected[remaining - 1];
            expected[remaining - 1] = expected[choice];
            expected[choice] = temporary;
            remaining -= 1;
            draw += 1;
        }
        let lattice = SquareLattice::new(
            geometry,
            SquareLatticeInitMethod::ShuffledValues { values, rng },
        )
        .unwrap();
        report.exact(
            "SquareLattice.ShuffledValues",
            lattice.data(),
            expected.as_slice(),
        );
    }
    report.finish();
}

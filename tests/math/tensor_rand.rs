use physics_in_parallel::prelude::basic::{
    DEFAULT_RANDOM_MAX_THREADS, RandType, RngConfig, RngConfigError, RngMethod, Scalar, Tensor,
    TensorRandError, TensorRandFiller,
};

fn rng(seed: u64, method: Option<RngMethod>) -> RngConfig {
    RngConfig::new(Some(seed), method)
}

fn values<T>(tensor: &Tensor<T>) -> Vec<T>
where
    T: Scalar + Copy,
{
    (0..tensor.size())
        .map(|i| tensor.get(&[i as isize]))
        .collect()
}

#[test]
fn seeded_filler_is_deterministic_across_refresh_sequences() {
    let kind = RandType::Uniform {
        low: -1.0,
        high: 1.0,
    };
    let mut a = Tensor::<f64>::empty(&[32]);
    let mut b = Tensor::<f64>::empty(&[32]);
    let mut filler_a = TensorRandFiller::new(kind, rng(12345, None));
    let mut filler_b = TensorRandFiller::new(kind, rng(12345, None));

    filler_a.refresh(&mut a);
    filler_b.refresh(&mut b);
    assert_eq!(values(&a), values(&b));

    filler_a.refresh(&mut a);
    filler_b.refresh(&mut b);
    assert_eq!(values(&a), values(&b));
}

#[test]
fn contiguous_slice_fill_matches_tensor_refresh() {
    let kind = RandType::Normal {
        mean: 0.0,
        std: 1.0,
    };
    let mut tensor = Tensor::<f64>::empty(&[32]);
    let mut slice = vec![0.0; 32];
    let mut tensor_filler = TensorRandFiller::new(kind, rng(91, None));
    let mut slice_filler = TensorRandFiller::new(kind, rng(91, None));

    tensor_filler.try_refresh(&mut tensor).unwrap();
    slice_filler.try_fill_slice(&mut slice).unwrap();

    assert_eq!(values(&tensor), slice);
}

#[test]
fn filler_worker_limit_is_instance_local_and_result_independent() {
    let kind = RandType::Normal {
        mean: 0.0,
        std: 1.0,
    };
    let mut serial = TensorRandFiller::new(kind, rng(12345, None))
        .with_max_threads(1)
        .unwrap();
    let mut parallel = TensorRandFiller::new(kind, rng(12345, None))
        .with_max_threads(4)
        .unwrap();
    let mut serial_values = vec![0.0; 32_768];
    let mut parallel_values = vec![0.0; 32_768];

    assert_eq!(
        TensorRandFiller::new(kind, rng(7, None)).max_threads(),
        DEFAULT_RANDOM_MAX_THREADS
    );
    assert_eq!(serial.max_threads(), 1);
    assert_eq!(parallel.max_threads(), 4);
    serial.try_fill_slice(&mut serial_values).unwrap();
    parallel.try_fill_slice(&mut parallel_values).unwrap();
    assert_eq!(serial_values, parallel_values);

    assert_eq!(
        parallel.set_max_threads(0),
        Err(TensorRandError::InvalidMaxThreads)
    );
    assert_eq!(parallel.max_threads(), 4);
}

#[test]
fn rng_kind_none_defaults_to_small_rng() {
    let kind = RandType::Uniform {
        low: -1.0,
        high: 1.0,
    };
    let mut default_tensor = Tensor::<f64>::empty(&[32]);
    let mut explicit_tensor = Tensor::<f64>::empty(&[32]);
    let mut default_filler = TensorRandFiller::new(kind, rng(12345, None));
    let mut explicit_filler = TensorRandFiller::new(kind, rng(12345, Some(RngMethod::SmallRng)));

    default_filler.refresh(&mut default_tensor);
    explicit_filler.refresh(&mut explicit_tensor);

    assert_eq!(
        default_filler.rng_config().method(),
        Some(RngMethod::SmallRng)
    );
    assert_eq!(
        explicit_filler.rng_config().method(),
        Some(RngMethod::SmallRng)
    );
    assert_eq!(values(&default_tensor), values(&explicit_tensor));
}

#[test]
fn selected_rng_kind_is_recorded_and_deterministic() {
    let kind = RandType::Uniform {
        low: -1.0,
        high: 1.0,
    };
    let mut a = Tensor::<f64>::empty(&[32]);
    let mut b = Tensor::<f64>::empty(&[32]);
    let mut filler_a = TensorRandFiller::new(kind, rng(12345, Some(RngMethod::Pcg64Mcg)));
    let mut filler_b = TensorRandFiller::new(kind, rng(12345, Some(RngMethod::Pcg64Mcg)));

    filler_a.refresh(&mut a);
    filler_b.refresh(&mut b);

    assert_eq!(filler_a.rng_config().method(), Some(RngMethod::Pcg64Mcg));
    assert_eq!(values(&a), values(&b));
}

#[test]
fn uniform_float_and_integer_ranges_are_respected() {
    let mut floats = Tensor::<f64>::empty(&[128]);
    let mut float_filler = TensorRandFiller::new(
        RandType::Uniform {
            low: 2.0,
            high: 3.0,
        },
        rng(7, None),
    );
    float_filler.refresh(&mut floats);
    assert!(values(&floats).iter().all(|&x| (2.0..3.0).contains(&x)));

    let mut ints = Tensor::<i64>::empty(&[128]);
    let mut int_filler =
        TensorRandFiller::new(RandType::UniformInt { low: -2, high: 2 }, rng(7, None));
    int_filler.refresh(&mut ints);
    assert!(values(&ints).iter().all(|&x| (-2..=2).contains(&x)));
}

#[test]
fn bernoulli_outputs_are_binary_for_float_and_integer_tensors() {
    let mut floats = Tensor::<f64>::empty(&[128]);
    let mut float_filler = TensorRandFiller::new(RandType::Bernoulli { p: 0.25 }, rng(11, None));
    float_filler.refresh(&mut floats);
    assert!(values(&floats).iter().all(|&x| x == 0.0 || x == 1.0));

    let mut ints = Tensor::<i64>::empty(&[128]);
    let mut int_filler = TensorRandFiller::new(RandType::Bernoulli { p: 0.25 }, rng(11, None));
    int_filler.refresh(&mut ints);
    assert!(values(&ints).iter().all(|&x| x == 0 || x == 1));
}

#[test]
fn fallible_constructor_rejects_incompatible_rng_method() {
    let err = TensorRandFiller::try_new(
        RandType::Bernoulli { p: 0.5 },
        RngConfig::new(Some(1), Some(RngMethod::IndexedSplitMix64)),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        TensorRandError::RngConfig(RngConfigError::UnsupportedMethod { .. })
    ));
}

#[test]
fn indexed_filler_replays_explicit_steps() {
    let config = RngConfig::new(Some(91), Some(RngMethod::IndexedSplitMix64));
    let filler = TensorRandFiller::try_new_indexed(
        RandType::Normal {
            mean: 0.0,
            std: 1.0,
        },
        config,
    )
    .unwrap();
    let mut first = vec![0.0; 257];
    let mut replay = vec![0.0; 257];
    filler.try_fill_slice_at(&mut first, 7, 11).unwrap();
    filler.try_fill_slice_at(&mut replay, 7, 11).unwrap();
    assert_eq!(first, replay);
}

#[test]
fn indexed_site_fill_is_exact_and_bounded() {
    fn fill() -> Vec<usize> {
        let filler = TensorRandFiller::try_new_indexed(
            RandType::Uniform {
                low: 0.0,
                high: 1.0,
            },
            rng(91, Some(RngMethod::IndexedSplitMix64)),
        )
        .unwrap();
        let mut sites = vec![0; 1_024];
        filler
            .try_fill_indices_at(&mut sites, 17, 7, 0xfeed)
            .unwrap();
        sites
    }

    let sites = fill();
    assert!(sites.iter().all(|site| *site < 17));
}

#[test]
fn indexed_site_fill_rejects_an_empty_range() {
    let filler = TensorRandFiller::try_new_indexed(
        RandType::Uniform {
            low: 0.0,
            high: 1.0,
        },
        rng(91, Some(RngMethod::IndexedSplitMix64)),
    )
    .unwrap();
    assert_eq!(
        filler.try_fill_indices_at(&mut [0; 1], 0, 0, 0),
        Err(TensorRandError::InvalidIndexUpperBound)
    );
}

#[test]
fn try_refresh_reports_invalid_distribution_parameters() {
    let mut floats = Tensor::<f64>::empty(&[4]);

    let mut uniform = TensorRandFiller::new(
        RandType::Uniform {
            low: 3.0,
            high: 2.0,
        },
        rng(1, None),
    );
    assert!(matches!(
        uniform.try_refresh(&mut floats),
        Err(TensorRandError::InvalidUniformBounds { .. })
    ));

    let mut normal = TensorRandFiller::new(
        RandType::Normal {
            mean: 0.0,
            std: -1.0,
        },
        rng(1, None),
    );
    assert!(matches!(
        normal.try_refresh(&mut floats),
        Err(TensorRandError::InvalidNormalStd { .. })
    ));

    let mut degenerate_normal = TensorRandFiller::new(
        RandType::Normal {
            mean: 2.5,
            std: 0.0,
        },
        rng(1, None),
    );
    degenerate_normal.try_refresh(&mut floats).unwrap();
    assert!(values(&floats).iter().all(|&value| value == 2.5));

    let mut bernoulli = TensorRandFiller::new(RandType::Bernoulli { p: 2.0 }, rng(1, None));
    assert!(matches!(
        bernoulli.try_refresh(&mut floats),
        Err(TensorRandError::InvalidBernoulliProbability { .. })
    ));
}

#[test]
fn try_refresh_reports_unsupported_distribution_for_scalar_type() {
    let mut floats = Tensor::<f64>::empty(&[4]);
    let mut filler = TensorRandFiller::new(RandType::UniformInt { low: 0, high: 1 }, rng(1, None));

    assert!(matches!(
        filler.try_refresh(&mut floats),
        Err(TensorRandError::UnsupportedDistribution { .. })
    ));
}

#[test]
fn try_refresh_reports_integer_bounds_out_of_range() {
    let mut unsigned = Tensor::<usize>::empty(&[4]);
    let mut filler = TensorRandFiller::new(RandType::UniformInt { low: -1, high: 1 }, rng(1, None));

    assert!(matches!(
        filler.try_refresh(&mut unsigned),
        Err(TensorRandError::IntegerBoundsOutOfRange { .. })
    ));
}

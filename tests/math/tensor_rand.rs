use physics_in_parallel::math::tensor::{
    RandType, RngKind, TensorRandError, TensorRandFiller, TensorTrait, dense,
};

fn values<T>(tensor: &dense::Tensor<T>) -> Vec<T>
where
    T: physics_in_parallel::math::Scalar + Copy,
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
    let mut a = dense::Tensor::<f64>::empty(&[32]);
    let mut b = dense::Tensor::<f64>::empty(&[32]);
    let mut filler_a = TensorRandFiller::new_with_seed(kind, Some(4), 12345);
    let mut filler_b = TensorRandFiller::new_with_seed(kind, Some(4), 12345);

    filler_a.refresh(&mut a);
    filler_b.refresh(&mut b);
    assert_eq!(values(&a), values(&b));

    filler_a.refresh(&mut a);
    filler_b.refresh(&mut b);
    assert_eq!(values(&a), values(&b));
}

#[test]
fn rng_kind_none_defaults_to_small_rng() {
    let kind = RandType::Uniform {
        low: -1.0,
        high: 1.0,
    };
    let mut default_tensor = dense::Tensor::<f64>::empty(&[32]);
    let mut explicit_tensor = dense::Tensor::<f64>::empty(&[32]);
    let mut default_filler =
        TensorRandFiller::new_with_seed_and_rng_kind(kind, Some(4), 12345, None);
    let mut explicit_filler =
        TensorRandFiller::new_with_seed_and_rng_kind(kind, Some(4), 12345, Some(RngKind::SmallRng));

    default_filler.refresh(&mut default_tensor);
    explicit_filler.refresh(&mut explicit_tensor);

    assert_eq!(default_filler.rng_kind(), RngKind::SmallRng);
    assert_eq!(explicit_filler.rng_kind(), RngKind::SmallRng);
    assert_eq!(values(&default_tensor), values(&explicit_tensor));
}

#[test]
fn selected_rng_kind_is_recorded_and_deterministic() {
    let kind = RandType::Uniform {
        low: -1.0,
        high: 1.0,
    };
    let mut a = dense::Tensor::<f64>::empty(&[32]);
    let mut b = dense::Tensor::<f64>::empty(&[32]);
    let mut filler_a =
        TensorRandFiller::new_with_seed_and_rng_kind(kind, Some(4), 12345, Some(RngKind::Pcg64Mcg));
    let mut filler_b =
        TensorRandFiller::new_with_seed_and_rng_kind(kind, Some(4), 12345, Some(RngKind::Pcg64Mcg));

    filler_a.refresh(&mut a);
    filler_b.refresh(&mut b);

    assert_eq!(filler_a.rng_kind(), RngKind::Pcg64Mcg);
    assert_eq!(values(&a), values(&b));
}

#[test]
fn uniform_float_and_integer_ranges_are_respected() {
    let mut floats = dense::Tensor::<f64>::empty(&[128]);
    let mut float_filler = TensorRandFiller::new_with_seed(
        RandType::Uniform {
            low: 2.0,
            high: 3.0,
        },
        Some(8),
        7,
    );
    float_filler.refresh(&mut floats);
    assert!(values(&floats).iter().all(|&x| (2.0..3.0).contains(&x)));

    let mut ints = dense::Tensor::<i64>::empty(&[128]);
    let mut int_filler =
        TensorRandFiller::new_with_seed(RandType::UniformInt { low: -2, high: 2 }, Some(8), 7);
    int_filler.refresh(&mut ints);
    assert!(values(&ints).iter().all(|&x| (-2..=2).contains(&x)));
}

#[test]
fn bernoulli_outputs_are_binary_for_float_and_integer_tensors() {
    let mut floats = dense::Tensor::<f64>::empty(&[128]);
    let mut float_filler =
        TensorRandFiller::new_with_seed(RandType::Bernoulli { p: 0.25 }, Some(4), 11);
    float_filler.refresh(&mut floats);
    assert!(values(&floats).iter().all(|&x| x == 0.0 || x == 1.0));

    let mut ints = dense::Tensor::<i64>::empty(&[128]);
    let mut int_filler =
        TensorRandFiller::new_with_seed(RandType::Bernoulli { p: 0.25 }, Some(4), 11);
    int_filler.refresh(&mut ints);
    assert!(values(&ints).iter().all(|&x| x == 0 || x == 1));
}

#[test]
fn fallible_constructors_reject_zero_rng_count() {
    let err = TensorRandFiller::try_new(RandType::Bernoulli { p: 0.5 }, Some(0)).unwrap_err();
    assert_eq!(err, TensorRandError::ZeroRngCount);

    let err = TensorRandFiller::try_new_with_seed(RandType::Bernoulli { p: 0.5 }, Some(0), 1)
        .unwrap_err();
    assert_eq!(err, TensorRandError::ZeroRngCount);
}

#[test]
fn try_refresh_reports_invalid_distribution_parameters() {
    let mut floats = dense::Tensor::<f64>::empty(&[4]);

    let mut uniform = TensorRandFiller::new_with_seed(
        RandType::Uniform {
            low: 3.0,
            high: 2.0,
        },
        Some(2),
        1,
    );
    assert!(matches!(
        uniform.try_refresh(&mut floats),
        Err(TensorRandError::InvalidUniformBounds { .. })
    ));

    let mut normal = TensorRandFiller::new_with_seed(
        RandType::Normal {
            mean: 0.0,
            std: 0.0,
        },
        Some(2),
        1,
    );
    assert!(matches!(
        normal.try_refresh(&mut floats),
        Err(TensorRandError::InvalidNormalStd { .. })
    ));

    let mut bernoulli = TensorRandFiller::new_with_seed(RandType::Bernoulli { p: 2.0 }, Some(2), 1);
    assert!(matches!(
        bernoulli.try_refresh(&mut floats),
        Err(TensorRandError::InvalidBernoulliProbability { .. })
    ));
}

#[test]
fn try_refresh_reports_unsupported_distribution_for_scalar_type() {
    let mut floats = dense::Tensor::<f64>::empty(&[4]);
    let mut filler =
        TensorRandFiller::new_with_seed(RandType::UniformInt { low: 0, high: 1 }, Some(2), 1);

    assert!(matches!(
        filler.try_refresh(&mut floats),
        Err(TensorRandError::UnsupportedDistribution { .. })
    ));
}

#[test]
fn try_refresh_reports_integer_bounds_out_of_range() {
    let mut unsigned = dense::Tensor::<usize>::empty(&[4]);
    let mut filler =
        TensorRandFiller::new_with_seed(RandType::UniformInt { low: -1, high: 1 }, Some(2), 1);

    assert!(matches!(
        filler.try_refresh(&mut unsigned),
        Err(TensorRandError::IntegerBoundsOutOfRange { .. })
    ));
}

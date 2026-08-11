use std::num::NonZeroUsize;

use physics_in_parallel::math::tensor::{
    RandType, TensorRandError, TensorRandFiller, TensorTrait, dense,
};
use physics_in_parallel::rng::{RngConfig, RngConfigError, RngMethod};

fn rng(seed: u64, method: Option<RngMethod>, streams: usize) -> RngConfig {
    RngConfig::new(Some(seed), method, NonZeroUsize::new(streams))
}

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
    let mut filler_a = TensorRandFiller::new(kind, rng(12345, None, 4));
    let mut filler_b = TensorRandFiller::new(kind, rng(12345, None, 4));

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
    let mut tensor = dense::Tensor::<f64>::empty(&[32]);
    let mut slice = vec![0.0; 32];
    let mut tensor_filler = TensorRandFiller::new(kind, rng(91, None, 4));
    let mut slice_filler = TensorRandFiller::new(kind, rng(91, None, 4));

    tensor_filler.try_refresh(&mut tensor).unwrap();
    slice_filler.try_fill_slice(&mut slice).unwrap();

    assert_eq!(values(&tensor), slice);
}

#[test]
fn rng_kind_none_defaults_to_small_rng() {
    let kind = RandType::Uniform {
        low: -1.0,
        high: 1.0,
    };
    let mut default_tensor = dense::Tensor::<f64>::empty(&[32]);
    let mut explicit_tensor = dense::Tensor::<f64>::empty(&[32]);
    let mut default_filler = TensorRandFiller::new(kind, rng(12345, None, 4));
    let mut explicit_filler = TensorRandFiller::new(kind, rng(12345, Some(RngMethod::SmallRng), 4));

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
    let mut a = dense::Tensor::<f64>::empty(&[32]);
    let mut b = dense::Tensor::<f64>::empty(&[32]);
    let mut filler_a = TensorRandFiller::new(kind, rng(12345, Some(RngMethod::Pcg64Mcg), 4));
    let mut filler_b = TensorRandFiller::new(kind, rng(12345, Some(RngMethod::Pcg64Mcg), 4));

    filler_a.refresh(&mut a);
    filler_b.refresh(&mut b);

    assert_eq!(filler_a.rng_config().method(), Some(RngMethod::Pcg64Mcg));
    assert_eq!(values(&a), values(&b));
}

#[test]
fn uniform_float_and_integer_ranges_are_respected() {
    let mut floats = dense::Tensor::<f64>::empty(&[128]);
    let mut float_filler = TensorRandFiller::new(
        RandType::Uniform {
            low: 2.0,
            high: 3.0,
        },
        rng(7, None, 8),
    );
    float_filler.refresh(&mut floats);
    assert!(values(&floats).iter().all(|&x| (2.0..3.0).contains(&x)));

    let mut ints = dense::Tensor::<i64>::empty(&[128]);
    let mut int_filler =
        TensorRandFiller::new(RandType::UniformInt { low: -2, high: 2 }, rng(7, None, 8));
    int_filler.refresh(&mut ints);
    assert!(values(&ints).iter().all(|&x| (-2..=2).contains(&x)));
}

#[test]
fn bernoulli_outputs_are_binary_for_float_and_integer_tensors() {
    let mut floats = dense::Tensor::<f64>::empty(&[128]);
    let mut float_filler = TensorRandFiller::new(RandType::Bernoulli { p: 0.25 }, rng(11, None, 4));
    float_filler.refresh(&mut floats);
    assert!(values(&floats).iter().all(|&x| x == 0.0 || x == 1.0));

    let mut ints = dense::Tensor::<i64>::empty(&[128]);
    let mut int_filler = TensorRandFiller::new(RandType::Bernoulli { p: 0.25 }, rng(11, None, 4));
    int_filler.refresh(&mut ints);
    assert!(values(&ints).iter().all(|&x| x == 0 || x == 1));
}

#[test]
fn fallible_constructor_rejects_incompatible_rng_method() {
    let err = TensorRandFiller::try_new(
        RandType::Bernoulli { p: 0.5 },
        RngConfig::new(Some(1), Some(RngMethod::IndexedSplitMix64), None),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        TensorRandError::RngConfig(RngConfigError::UnsupportedMethod { .. })
    ));
}

#[test]
fn try_refresh_reports_invalid_distribution_parameters() {
    let mut floats = dense::Tensor::<f64>::empty(&[4]);

    let mut uniform = TensorRandFiller::new(
        RandType::Uniform {
            low: 3.0,
            high: 2.0,
        },
        rng(1, None, 2),
    );
    assert!(matches!(
        uniform.try_refresh(&mut floats),
        Err(TensorRandError::InvalidUniformBounds { .. })
    ));

    let mut normal = TensorRandFiller::new(
        RandType::Normal {
            mean: 0.0,
            std: 0.0,
        },
        rng(1, None, 2),
    );
    assert!(matches!(
        normal.try_refresh(&mut floats),
        Err(TensorRandError::InvalidNormalStd { .. })
    ));

    let mut bernoulli = TensorRandFiller::new(RandType::Bernoulli { p: 2.0 }, rng(1, None, 2));
    assert!(matches!(
        bernoulli.try_refresh(&mut floats),
        Err(TensorRandError::InvalidBernoulliProbability { .. })
    ));
}

#[test]
fn try_refresh_reports_unsupported_distribution_for_scalar_type() {
    let mut floats = dense::Tensor::<f64>::empty(&[4]);
    let mut filler =
        TensorRandFiller::new(RandType::UniformInt { low: 0, high: 1 }, rng(1, None, 2));

    assert!(matches!(
        filler.try_refresh(&mut floats),
        Err(TensorRandError::UnsupportedDistribution { .. })
    ));
}

#[test]
fn try_refresh_reports_integer_bounds_out_of_range() {
    let mut unsigned = dense::Tensor::<usize>::empty(&[4]);
    let mut filler =
        TensorRandFiller::new(RandType::UniformInt { low: -1, high: 1 }, rng(1, None, 2));

    assert!(matches!(
        filler.try_refresh(&mut unsigned),
        Err(TensorRandError::IntegerBoundsOutOfRange { .. })
    ));
}

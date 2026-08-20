use physics_in_parallel::prelude::basic::{
    RngConfig, VectorList, VectorSamplingError, VectorSamplingMethod, sample_vectors,
};

#[test]
fn uniform_centered_sampling_fills_vector_list_inside_box() {
    let mut positions = VectorList::<f64>::empty(2, 256);

    sample_vectors(
        &mut positions,
        VectorSamplingMethod::UniformCentered {
            box_size: &[2.0, 4.0],
        },
        RngConfig::default(),
    )
    .unwrap();

    for i in 0..positions.num_vectors() {
        let row = positions.vector(i as isize);
        assert!(row[0] >= -1.0 && row[0] <= 1.0);
        assert!(row[1] >= -2.0 && row[1] <= 2.0);
    }
}

#[test]
fn jittered_lattice_without_noise_places_regular_points() {
    let mut positions = VectorList::<f64>::empty(2, 4);

    sample_vectors(
        &mut positions,
        VectorSamplingMethod::JitteredLattice {
            spacings: &[10.0, 100.0],
            sigmas: &[0.0, 0.0],
        },
        RngConfig::default(),
    )
    .unwrap();

    assert_eq!(positions.vector(0), [0.0, 0.0].as_slice());
    assert_eq!(positions.vector(1), [10.0, 0.0].as_slice());
    assert_eq!(positions.vector(2), [0.0, 100.0].as_slice());
    assert_eq!(positions.vector(3), [10.0, 100.0].as_slice());
}

#[test]
fn vector_sampling_rejects_bad_parameters() {
    let mut positions = VectorList::<f64>::empty(2, 4);

    assert_eq!(
        sample_vectors(
            &mut positions,
            VectorSamplingMethod::UniformCentered { box_size: &[1.0] },
            RngConfig::default(),
        )
        .unwrap_err(),
        VectorSamplingError::InvalidParameterLength {
            parameter: "box_size",
            expected: 2,
            got: 1,
        }
    );

    match sample_vectors(
        &mut positions,
        VectorSamplingMethod::JitteredLattice {
            spacings: &[1.0, f64::NAN],
            sigmas: &[0.0, 0.0],
        },
        RngConfig::default(),
    )
    .unwrap_err()
    {
        VectorSamplingError::InvalidParameterValue {
            parameter,
            index,
            value,
            rule,
        } => {
            assert_eq!(parameter, "spacings");
            assert_eq!(index, 1);
            assert!(value.is_nan());
            assert_eq!(rule, "finite and non-negative");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn uniform_and_gaussian_per_axis_sampling_cover_generic_vectors() {
    let mut vectors = VectorList::<f64>::empty(2, 64);

    sample_vectors(
        &mut vectors,
        VectorSamplingMethod::Uniform {
            low: -2.0,
            high: 3.0,
        },
        RngConfig::default(),
    )
    .unwrap();

    for i in 0..vectors.num_vectors() {
        for &x in vectors.vector(i as isize) {
            assert!((-2.0..3.0).contains(&x));
        }
    }

    sample_vectors(
        &mut vectors,
        VectorSamplingMethod::GaussianPerAxis {
            mean: &[5.0, -5.0],
            std: &[0.0, 0.0],
        },
        RngConfig::default(),
    )
    .unwrap();

    for i in 0..vectors.num_vectors() {
        assert_eq!(vectors.vector(i as isize), [5.0, -5.0].as_slice());
    }

    assert_eq!(
        sample_vectors(
            &mut vectors,
            VectorSamplingMethod::Uniform {
                low: 1.0,
                high: 1.0,
            },
            RngConfig::default(),
        )
        .unwrap_err(),
        VectorSamplingError::InvalidUniformBounds {
            low: 1.0,
            high: 1.0,
        }
    );
}

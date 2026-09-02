use std::error::Error;

use physics_in_parallel::prelude::advanced::{InteractionError, NeighborListError};
use physics_in_parallel::prelude::basic::{
    BoundaryError, IndexedRng, RngConfig, RngConfigError, RngMethod, TensorRandError,
    VectorSamplingError,
};
use physics_in_parallel::prelude::models::*;

fn assert_error_contract<T: Error + Send + Sync + 'static>() {}

#[test]
fn public_model_errors_satisfy_standard_error_contract() {
    assert_error_contract::<AttrsError>();
    assert_error_contract::<InteractionError>();
    assert_error_contract::<NeighborListError>();
    assert_error_contract::<BoundaryError>();
    assert_error_contract::<VectorSamplingError>();
    assert_error_contract::<RngConfigError>();
    assert_error_contract::<TensorRandError>();
    assert_error_contract::<MassiveParticlesError>();
    assert_error_contract::<IntegratorError>();
    assert_error_contract::<ThermostatError>();
    assert_error_contract::<ObserveError>();
    assert_error_contract::<ParticleBoundaryError>();
    assert_error_contract::<ParticleNeighborListError>();
    assert_error_contract::<SpringLawError>();
    assert_error_contract::<SpringNetworkError>();
    assert_error_contract::<PowerLawError>();
    assert_error_contract::<PowerLawNetworkError>();
}

#[test]
fn nested_sampling_errors_preserve_the_complete_source_chain() {
    let rng_error = IndexedRng::new(RngConfig::new(Some(7), Some(RngMethod::Pcg64)))
        .expect_err("PCG is unsupported by indexed randomness");
    let error = MassiveParticlesError::Sampling(VectorSamplingError::Rng(
        TensorRandError::RngConfig(rng_error),
    ));

    assert_eq!(
        error.to_string(),
        "particle sampling error: vector sampling RNG error: RNG method `pcg64` is not supported by IndexedRng"
    );

    let sampling = error.source().expect("sampling source");
    let tensor = sampling.source().expect("tensor RNG source");
    let config = tensor.source().expect("RNG config source");

    assert!(
        sampling
            .to_string()
            .starts_with("vector sampling RNG error:")
    );
    assert_eq!(
        config.to_string(),
        "RNG method `pcg64` is not supported by IndexedRng"
    );
    assert!(config.source().is_none());
}

#[test]
fn model_wrappers_add_context_without_hiding_the_source() {
    let error =
        SpringNetworkError::Interaction(InteractionError::Attrs(AttrsError::UnknownLabel {
            label: "r".to_string(),
        }));

    assert_eq!(
        error.to_string(),
        "spring-network topology error: interaction attribute error: unknown attribute label `r`"
    );

    let interaction = error.source().expect("interaction source");
    let attrs = interaction.source().expect("attribute source");
    assert_eq!(attrs.to_string(), "unknown attribute label `r`");
}

#[test]
fn validation_errors_report_relevant_values() {
    assert_eq!(
        IntegratorError::InvalidDt { dt: -0.5 }.to_string(),
        "integration time step must be finite and positive; got -0.5"
    );
    assert_eq!(
        BoundaryError::InvalidVectorDimension {
            label: "position",
            expected: 3,
            got: 2,
        }
        .to_string(),
        "position vector has dimension 2; expected 3"
    );
}

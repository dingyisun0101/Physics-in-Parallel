use physics_in_parallel::prelude::basic::{
    PairGenerator, PairingMethod, ResolvedRng, RngMethod, SourceMode, SquareLattice,
    SquareLatticeGeometry, SquareLatticeInitMethod, StorageKind, VectorList, VectorSamplingMethod,
    sample_vectors,
};

fn indexed_rng(seed: u64) -> ResolvedRng {
    ResolvedRng::new(seed, RngMethod::IndexedSplitMix64)
}

#[test]
fn lattice_serde_preserves_geometry_values_and_rng_provenance() {
    let geometry = SquareLatticeGeometry::new(
        &[2, 3],
        physics_in_parallel::prelude::basic::BoundaryCondition::Periodic,
        Some(&[0.5, 2.0]),
    )
    .unwrap();
    let lattice = SquareLattice::new(
        geometry,
        SquareLatticeInitMethod::ShuffledValues {
            values: (0_i32..6).collect(),
            rng: indexed_rng(17),
        },
    )
    .unwrap();

    let json = serde_json::to_string(&lattice).unwrap();
    let restored: SquareLattice<i32> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.geometry(), lattice.geometry());
    assert_eq!(restored.data(), lattice.data());
    assert_eq!(
        restored.initialization_resolved_rng(),
        Some(indexed_rng(17))
    );
}

#[test]
fn pair_generation_uses_resolved_rng_and_universal_outputs() {
    let method = PairingMethod::IndependentUniform;
    let mut pairs = PairGenerator::new(&[4, 5], method, 8, indexed_rng(11)).unwrap();
    pairs.refresh_at(9);

    assert_eq!(pairs.generated_sweep(), Some(9));
    assert_eq!(pairs.resolved_rng(), indexed_rng(11));
    assert_eq!(pairs.sources().shape(), [8, 2]);
    assert_eq!(pairs.sources().storage_kind(), StorageKind::Dense);
    assert_eq!(pairs.method(), method);

    let kernel_method = PairingMethod::Kernel {
        kernel: physics_in_parallel::prelude::basic::KernelType::NearestNeighbor { d: 2 },
        sources: SourceMode::Origin,
    };
    assert!(PairGenerator::new(&[4, 5], kernel_method, 8, indexed_rng(12)).is_ok());
}

#[test]
fn sampling_preserves_the_callers_storage_representation() {
    let mut values = VectorList::<f64>::zeros(2, 4).unwrap();
    assert_eq!(values.storage_kind(), StorageKind::Sparse);
    sample_vectors(
        &mut values,
        VectorSamplingMethod::Uniform {
            low: 1.0,
            high: 2.0,
        },
        indexed_rng(23),
    )
    .unwrap();

    assert_eq!(values.storage_kind(), StorageKind::Sparse);
    assert!(values.values().all(|value| (1.0..2.0).contains(&value)));
}

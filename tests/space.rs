use physics_in_parallel::prelude::basic::{
    Backend, PairGenerator, PairingMethod, ResolvedRng, RngMethod, SourceMode, SquareLattice,
    SquareLatticeGeometry, SquareLatticeInitMethod, VectorList, VectorSamplingMethod,
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
    assert_eq!(pairs.sources().backend(), Backend::Dense);
    assert_eq!(pairs.method(), method);

    let kernel_method = PairingMethod::Kernel {
        kernel: physics_in_parallel::prelude::basic::KernelType::NearestNeighbor { dimension: 2 },
        sources: SourceMode::Origin,
    };
    assert!(PairGenerator::new(&[4, 5], kernel_method, 8, indexed_rng(12)).is_ok());
}

#[test]
fn sampling_preserves_the_callers_backend() {
    let mut values = VectorList::<f64>::zeros(2, 4, Backend::Sparse).unwrap();
    assert_eq!(values.backend(), Backend::Sparse);
    sample_vectors(
        &mut values,
        VectorSamplingMethod::Uniform {
            low: 1.0,
            high: 2.0,
        },
        indexed_rng(23),
    )
    .unwrap();

    assert_eq!(values.backend(), Backend::Sparse);
    assert!(values.values().all(|value| (1.0..2.0).contains(&value)));
}

#[test]
fn boundary_geometry_and_reflection_parity() {
    use physics_in_parallel::space::{ContinuousBoundary, PeriodicBox, ReflectBox};
    assert!(PeriodicBox::new(&[], &[]).is_err());
    assert!(PeriodicBox::new(&[-f64::MAX], &[f64::MAX]).is_err());
    assert!(ReflectBox::new(&[0.0], &[f64::MAX]).is_err());
    let clamp = physics_in_parallel::space::ClampBox::new(&[-f64::MAX], &[f64::MAX]).unwrap();
    let mut position = [0.0];
    clamp.apply_position(&mut position).unwrap();
    assert_eq!(position, [0.0]);
    let boundary = ReflectBox::new(&[0.0], &[1.0]).unwrap();
    for (input, expected, sign) in [
        (-3.0, 1.0, -1.0),
        (-2.0, 0.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-0.25, 0.25, -1.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.25, 0.75, -1.0),
        (2.0, 0.0, -1.0),
        (3.0, 1.0, 1.0),
        (4.0, 0.0, -1.0),
    ] {
        let mut r = [input];
        let mut v = [2.0];
        boundary.apply_position_velocity(&mut r, &mut v).unwrap();
        assert_eq!(r, [expected]);
        assert_eq!(v, [2.0 * sign]);
        let mut position_only = [input];
        boundary.apply_position(&mut position_only).unwrap();
        assert_eq!(position_only, r);
    }
    let periodic = PeriodicBox::new(&[-1e308], &[-0.5e308]).unwrap();
    let mut distant = [f64::MAX];
    periodic.apply_position(&mut distant).unwrap();
    assert!(distant[0].is_finite() && (-1e308..-0.5e308).contains(&distant[0]));
}

#[test]
fn contiguous_laplacian_matches_neighbor_reference() {
    use physics_in_parallel::prelude::basic::BoundaryCondition;
    for shape in [
        &[1][..],
        &[2],
        &[7],
        &[1, 3],
        &[4, 5],
        &[3, 2, 4],
        &[2, 1, 3, 2],
    ] {
        for boundary in [
            BoundaryCondition::Periodic,
            BoundaryCondition::Reflective,
            BoundaryCondition::Neumann,
        ] {
            let spacing: Vec<_> = (0..shape.len()).map(|axis| 0.5 + axis as f64).collect();
            let geometry = SquareLatticeGeometry::new(shape, boundary, Some(&spacing)).unwrap();
            for components in [1, 2, 3, 8, 17] {
                let input: Vec<_> = (0..geometry.num_sites() * components)
                    .map(|i| (i as f64 * 0.13).sin())
                    .collect();
                let mut output = vec![99.0; input.len()];
                geometry.laplacian(&input, components, &mut output).unwrap();
                for site in 0..geometry.num_sites() {
                    for component in 0..components {
                        let mut expected = 0.0;
                        for (axis, spacing) in spacing.iter().enumerate() {
                            let plus = geometry.neighbor(site as isize, axis, 1).unwrap();
                            let minus = geometry.neighbor(site as isize, axis, -1).unwrap();
                            expected += (input[plus * components + component]
                                + input[minus * components + component]
                                - 2.0 * input[site * components + component])
                                * (1.0 / (spacing * spacing));
                        }
                        assert_eq!(
                            output[site * components + component].to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        }
    }
}

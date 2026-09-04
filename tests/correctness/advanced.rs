use super::support::*;
use physics_in_parallel::prelude::advanced::*;
use physics_in_parallel::prelude::basic::*;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn raw_storage_and_heterogeneous_attributes_match_plain_vectors() {
    let mut report = Report::default();
    for backend in BACKENDS {
        let values = data(21, 3);
        let mut t = Tensor::from_values(&[3, 7], backend, values.clone()).unwrap();
        let mut m = Matrix::from_values(3, 7, backend, values.clone()).unwrap();
        let mut v = VectorList::from_values(7, 3, backend, values.clone()).unwrap();
        for raw in [&mut t as &mut dyn RawStorage<f64>, &mut m, &mut v] {
            let mut expected = values.clone();
            for (i, value) in [(0, 1.75), (7, 0.0), (20, -3.5)] {
                assert!(raw.set_value_at_index(i, value));
                expected[i] = value;
            }
            assert!(!raw.set_value_at_index(21, 999.0));
            assert_eq!(raw.value_at_index(21), None);
            report.close(
                "RawStorage.flat_access",
                &(0..21)
                    .map(|i| raw.value_at_index(i).unwrap())
                    .collect::<Vec<_>>(),
                &expected,
                0.0,
                0.0,
            );
            if let Some(slice) = raw.dense_values_mut() {
                slice[4] = 8.0;
                expected[4] = 8.0;
            }
            report.exact(
                "RawStorage.dense_slice availability",
                raw.dense_values().is_some(),
                backend == Backend::Dense,
            );
            let entries: BTreeMap<_, _> = raw.stored_entries().into_iter().collect();
            let mut visited = BTreeMap::new();
            raw.for_each_stored_entry(&mut |i, v| {
                visited.insert(i, v);
            });
            let reference: BTreeMap<_, _> = expected
                .iter()
                .enumerate()
                .filter(|(_, v)| backend == Backend::Dense || **v != 0.0)
                .map(|(i, &v)| (i, v))
                .collect();
            report.exact("RawStorage.entries", entries, reference.clone());
            report.exact("RawStorage.visitor", visited, reference);
        }
        let mut core = AttrsCore::empty();
        core.insert(
            "r",
            VectorList::from_values(7, 3, backend, values.clone()).unwrap(),
        )
        .unwrap();
        core.allocate::<f64>("v", 7, 3).unwrap();
        let id = core.id_of("r").unwrap();
        report.close(
            "AttrsCore.get/get_by_id",
            &core
                .get_by_id::<f64>(id)
                .unwrap()
                .values()
                .collect::<Vec<_>>(),
            &values,
            0.0,
            0.0,
        );
        core.set_vector_of("v", 1, &[1.0; 7]).unwrap();
        report.close(
            "AttrsCore.vector access",
            &core.vector_of::<f64>("v", 1).unwrap(),
            &[1.0; 7],
            0.0,
            0.0,
        );
        core.rename("v", "velocity").unwrap();
        let (r, v) = core.get_two_mut::<f64>("r", "velocity").unwrap();
        r.set(0, 0, 2.0).unwrap();
        v.set(0, 0, 3.0).unwrap();
        let meta = AttrsMeta::new(13, "reference", "plain vectors");
        let object = physics_in_parallel::models::PhysObj::from_raw_parts(meta, core);
        report.exact("PhysObjAdvanced.metadata", object.id(), 13);
        report.close(
            "PhysObjAdvanced.attributes",
            &object.attribute_vector::<f64>("velocity", 0).unwrap(),
            &[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0.0,
            0.0,
        );
        let mut restored = object.clone();
        restored.attributes_mut().remove("velocity").unwrap();
        report.exact(
            "AttrsCore.remove",
            restored.has_attribute("velocity"),
            false,
        );
    }
    report.finish();
}

#[test]
fn structured_matrices_match_explicit_dense_algebra() {
    let mut report = Report::default();
    macro_rules! structured {
        ($kind:ident,$support:expr,$mirror:expr) => {{
            for n in [1, 3, 17] {
                let mut a = $kind::<f64>::zeros(n, n);
                let mut expected = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        if ($support)(i, j) {
                            let value = 0.25 + ((i * 7 + j * 11) % 19) as f64 / 7.0;
                            a.set(i as isize, j as isize, value);
                            expected[i * n + j] = value;
                            if $mirror != 0 && i != j {
                                expected[j * n + i] = value * $mirror as f64;
                            }
                        }
                    }
                }
                let dense = a.to_dense_matrix();
                let absolute = dense.abs();
                report.close(
                    "advanced dense.abs",
                    &(0..n * n)
                        .map(|i| absolute.get_flat(i as isize))
                        .collect::<Vec<_>>(),
                    &expected.iter().map(|v| v.abs()).collect::<Vec<_>>(),
                    ATOL,
                    RTOL,
                );
                report.close(
                    concat!(stringify!($kind), ".get/to_dense"),
                    &(0..n * n)
                        .map(|i| dense.get_flat(i as isize))
                        .collect::<Vec<_>>(),
                    &expected,
                    ATOL,
                    RTOL,
                );
                assert_eq!(a.shape(), [n, n]);
                assert_eq!(a.size(), n * n);
                report.close(
                    concat!(stringify!($kind), ".wrapped_get"),
                    &[a.get(-1, -1), a.get_flat(-1)],
                    &[expected[n * n - 1]; 2],
                    ATOL,
                    RTOL,
                );
                let input = data(3 * n, 7);
                let batch: Vec<_> = input
                    .chunks(n)
                    .flat_map(|x| matmul(&expected, x, n, n, 1))
                    .collect();
                let mut out = vec![99.0; 3 * n];
                a.mul_vectors_into(&input, &mut out).unwrap();
                report.close(
                    concat!(stringify!($kind), ".mul_vectors_into"),
                    &out,
                    &batch,
                    ATOL,
                    RTOL,
                );
                a.mul_vector_into(&input[..n], &mut out[..n]).unwrap();
                report.close(
                    concat!(stringify!($kind), ".mul_vector_into"),
                    &out[..n],
                    &batch[..n],
                    ATOL,
                    RTOL,
                );
                let trace = (0..n).map(|i| expected[i * n + i]).sum();
                report.close(
                    concat!(stringify!($kind), ".trace/max_abs"),
                    &[a.trace(), a.max_abs_real()],
                    &[trace, expected.iter().map(|v| v.abs()).fold(0.0, f64::max)],
                    ATOL,
                    RTOL,
                );
                for (label, result, reference) in [
                    (
                        "add",
                        a.add(&a),
                        expected.iter().map(|v| 2.0 * v).collect::<Vec<_>>(),
                    ),
                    ("sub", a.sub(&a), vec![0.0; n * n]),
                    (
                        "scalar_mul",
                        a.scalar_mul(-1.5),
                        expected.iter().map(|v| -1.5 * v).collect(),
                    ),
                ] {
                    report.close(
                        &format!("{}.{label}", stringify!($kind)),
                        &(0..n * n)
                            .map(|i| result.get_flat(i as isize))
                            .collect::<Vec<_>>(),
                        &reference,
                        ATOL,
                        RTOL,
                    );
                }
                if $mirror != -1 {
                    for (label, result, reference) in [
                        (
                            "elem_mul",
                            a.elem_mul(&a),
                            expected.iter().map(|v| v * v).collect::<Vec<_>>(),
                        ),
                        (
                            "matmul",
                            a.matmul(&a),
                            matmul(&expected, &expected, n, n, n),
                        ),
                    ] {
                        report.close(
                            &format!("{}.{label}", stringify!($kind)),
                            &(0..n * n)
                                .map(|i| result.get_flat(i as isize))
                                .collect::<Vec<_>>(),
                            &reference,
                            ATOL,
                            RTOL,
                        );
                    }
                }
                if $mirror != 0 || stringify!($kind) == "DiagonalMatrix" {
                    for result in [a.transpose(), a.hermitian_transpose()] {
                        report.close(
                            concat!(stringify!($kind), ".transpose/hermitian_transpose"),
                            &(0..n * n)
                                .map(|i| result.get_flat(i as isize))
                                .collect::<Vec<_>>(),
                            &transpose(&expected, n, n),
                            ATOL,
                            RTOL,
                        );
                    }
                }
                if stringify!($kind) == "SymmetricMatrix" {
                    let quotient = a.elem_div(&a);
                    report.close(
                        "SymmetricMatrix.elem_div",
                        &(0..n * n)
                            .map(|i| quotient.get_flat(i as isize))
                            .collect::<Vec<_>>(),
                        &vec![1.0; n * n],
                        ATOL,
                        RTOL,
                    );
                }
                // Dense-returning methods retain the full answer even when the
                // result cannot be represented by the original structure.
                for (label, result, reference) in [
                    (
                        "add_to_dense",
                        a.add_to_dense(&a),
                        expected.iter().map(|v| 2.0 * v).collect::<Vec<_>>(),
                    ),
                    ("sub_to_dense", a.sub_to_dense(&a), vec![0.0; n * n]),
                    (
                        "elem_mul_to_dense",
                        a.elem_mul_to_dense(&a),
                        expected.iter().map(|v| v * v).collect(),
                    ),
                    (
                        "elem_div_to_dense",
                        a.elem_div_to_dense(&a),
                        expected
                            .iter()
                            .map(|&v| if v == 0.0 { f64::NAN } else { 1.0 })
                            .collect(),
                    ),
                    (
                        "scalar_mul_to_dense",
                        a.scalar_mul_to_dense(-1.5),
                        expected.iter().map(|v| -1.5 * v).collect(),
                    ),
                    (
                        "transpose_to_dense",
                        a.transpose_to_dense(),
                        transpose(&expected, n, n),
                    ),
                    (
                        "hermitian_transpose_to_dense",
                        a.hermitian_transpose_to_dense(),
                        transpose(&expected, n, n),
                    ),
                    (
                        "matmul_to_dense",
                        a.matmul_to_dense(&a),
                        matmul(&expected, &expected, n, n, n),
                    ),
                ] {
                    report.close(
                        &format!("{}.{label}", stringify!($kind)),
                        &(0..n * n)
                            .map(|i| result.get_flat(i as isize))
                            .collect::<Vec<_>>(),
                        &reference,
                        ATOL,
                        RTOL,
                    );
                }
                let sparse = dense.to_sparse();
                let roundtrip = sparse.to_dense();
                report.close(
                    "advanced dense/sparse conversion",
                    &(0..n * n)
                        .map(|i| roundtrip.get_flat(i as isize))
                        .collect::<Vec<_>>(),
                    &expected,
                    0.0,
                    0.0,
                );
                let cast = dense.try_cast_to::<f32>().unwrap();
                report.close(
                    "advanced cast",
                    &(0..n * n)
                        .map(|i| cast.get_flat(i as isize) as f64)
                        .collect::<Vec<_>>(),
                    &expected,
                    1e-6,
                    1e-6,
                );
            }
        }};
    }
    structured!(DiagonalMatrix, |i, j| i == j, 0);
    structured!(SymmetricMatrix, |i, j| i <= j, 1);
    structured!(AntiSymmetricMatrix, |i, j| i < j, -1);
    structured!(UpperTriangularMatrix, |i, j| i <= j, 0);
    structured!(StrictUpperTriangularMatrix, |i, j| i < j, 0);
    structured!(LowerTriangularMatrix, |i, j| i >= j, 0);
    structured!(StrictLowerTriangularMatrix, |i, j| i > j, 0);
    for n in [31, 32, 127, 128, 16403] {
        let mut diagonal = DiagonalMatrix::<f64>::zeros(n, n);
        diagonal.fill(1.75);
        let input = data(n, 3);
        let mut out = vec![0.0; n];
        diagonal.mul_vector_into(&input, &mut out).unwrap();
        report.close(
            "DiagonalMatrix SIMD lengths",
            &out,
            &input.iter().map(|v| 1.75 * v).collect::<Vec<_>>(),
            ATOL,
            RTOL,
        );
    }
    report.finish();
}

#[test]
fn weighted_selection_matches_expanded_ticket_list_after_updates() {
    let mut report = Report::default();
    let mut weights: Vec<_> = (0..67).map(|i| i % 7).collect();
    let mut index = DynamicWeightedIndex::new(&weights).unwrap();
    for step in 0..100 {
        let i = (step * 37) % weights.len();
        weights[i] = (step * 11) % 13;
        index.set_weight(i, weights[i]).unwrap();
        let tickets: Vec<_> = weights
            .iter()
            .enumerate()
            .flat_map(|(i, &n)| std::iter::repeat_n(i, n))
            .collect();
        report.exact(
            "DynamicWeightedIndex.select",
            (0..=tickets.len())
                .map(|i| index.select(i))
                .collect::<Vec<_>>(),
            tickets
                .iter()
                .copied()
                .map(Some)
                .chain([None])
                .collect::<Vec<_>>(),
        );
        for excluded in [0, i, 66] {
            let filtered: Vec<_> = tickets.iter().copied().filter(|&j| j != excluded).collect();
            report.exact(
                "DynamicWeightedIndex.select_excluding",
                (0..=filtered.len())
                    .map(|i| index.select_excluding(i, excluded))
                    .collect::<Vec<_>>(),
                filtered
                    .into_iter()
                    .map(Some)
                    .chain([None])
                    .collect::<Vec<_>>(),
            );
        }
        for (i, &w) in weights.iter().enumerate() {
            assert_eq!(index.weight(i), Some(w));
        }
    }
    report.finish();
}

#[test]
fn interaction_engine_matches_map_through_mutations() {
    let mut report = Report::default();
    for order in [InteractionOrder::Ordered, InteractionOrder::Unordered] {
        let mut engine = Interaction::<f64>::new(19, order);
        let mut reference = BTreeMap::new();
        for step in 0..200 {
            let nodes = vec![step % 19, (step % 19 + 1 + step % 17) % 19];
            let mut key = nodes.clone();
            if order == InteractionOrder::Unordered {
                key.sort_unstable();
            }
            if step % 4 == 0 {
                engine.remove(&nodes).unwrap();
                reference.remove(&key);
            } else {
                let value = step as f64 / 7.0;
                engine.set(&nodes, value).unwrap();
                reference.insert(key.clone(), value);
            }
            report.exact(
                "Interaction.get/contains",
                engine.get(&nodes).unwrap().copied(),
                reference.get(&key).copied(),
            );
            let actual: BTreeMap<_, _> = engine
                .iter()
                .map(|(_, nodes, &v)| (nodes.nodes.to_vec(), v))
                .collect();
            report.exact("Interaction.iter", actual, reference.clone());
            assert_eq!(engine.len(), reference.len());
        }
        engine.par_for_each_payload_mut(|_, v| *v *= 2.0);
        for v in reference.values_mut() {
            *v *= 2.0;
        }
        let removed = engine.prune_n_objects(9);
        reference.retain(|key, _| key.iter().all(|&i| i < 9));
        assert!(removed.iter().all(|(_, v)| v.is_finite()));
        report.exact(
            "Interaction.prune/parallel payload",
            engine
                .iter()
                .map(|(_, n, &v)| (n.nodes.to_vec(), v))
                .collect::<BTreeMap<_, _>>(),
            reference,
        );
        engine.clear();
        assert!(engine.is_empty());
        let mut topology = InteractionTopology::with_order(19, order);
        let mut expected = BTreeSet::new();
        for i in 0..15 {
            let mut nodes = vec![i, (i + 5) % 19, (i + 11) % 19];
            topology.add(&nodes).unwrap();
            if order == InteractionOrder::Unordered {
                nodes.sort_unstable();
            }
            expected.insert(nodes);
        }
        report.exact(
            "InteractionTopology higher arity",
            topology
                .iter()
                .map(|(_, n)| n.nodes.to_vec())
                .collect::<BTreeSet<_>>(),
            expected,
        );
    }
    report.finish();
}

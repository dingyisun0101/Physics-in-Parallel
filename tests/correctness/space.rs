use super::support::*;
use physics_in_parallel::prelude::advanced::*;
use physics_in_parallel::prelude::basic::*;
use physics_in_parallel::prelude::models::*;
use std::collections::BTreeSet;

fn normalize(mut x: isize, n: usize, boundary: BoundaryCondition) -> usize {
    if n == 1 {
        return 0;
    }
    let high = n as isize - 1;
    match boundary {
        BoundaryCondition::Periodic => {
            while x < 0 {
                x += n as isize;
            }
            while x > high {
                x -= n as isize;
            }
        }
        BoundaryCondition::Reflective => {
            while x < 0 || x > high {
                if x < 0 {
                    x = -x;
                }
                if x > high {
                    x = 2 * high - x;
                }
            }
        }
        BoundaryCondition::Neumann => {
            if x < 0 {
                x = 0;
            }
            if x > high {
                x = high;
            }
        }
    }
    x as usize
}

#[test]
fn lattice_coordinates_neighbors_and_laplacian_against_grid_loops() {
    let mut report = Report::default();
    for shape in [
        vec![1],
        vec![7],
        vec![1, 3],
        vec![4, 5],
        vec![3, 2, 4],
        vec![2, 1, 3, 2],
        vec![129, 131],
    ] {
        for boundary in [
            BoundaryCondition::Periodic,
            BoundaryCondition::Reflective,
            BoundaryCondition::Neumann,
        ] {
            let spacing: Vec<_> = (0..shape.len()).map(|i| 0.75 + i as f64 * 0.5).collect();
            let geometry = SquareLatticeGeometry::new(&shape, boundary, Some(&spacing)).unwrap();
            let n = shape.iter().product();
            for index in [-3 * n as isize, -1, 0, n as isize - 1, 3 * n as isize + 1] {
                let expected = coords(normalize(index, n, boundary), &shape)
                    .into_iter()
                    .map(|x| x as isize)
                    .collect::<Vec<_>>();
                report.exact(
                    "SquareLatticeGeometry.coordinate",
                    geometry.coordinate(index),
                    expected,
                );
            }
            for index in 0..n {
                let coordinate = coords(index, &shape);
                for axis in 0..shape.len() {
                    for offset in [-3, -1, 0, 1, 4] {
                        let mut target = coordinate.clone();
                        target[axis] =
                            normalize(coordinate[axis] as isize + offset, shape[axis], boundary);
                        report.exact(
                            "SquareLatticeGeometry.neighbor",
                            geometry.neighbor(index as isize, axis, offset),
                            Some(flat(&target, &shape)),
                        );
                        let mut signed: Vec<_> = coordinate.iter().map(|&x| x as isize).collect();
                        signed[axis] += offset;
                        report.exact(
                            "SquareLatticeGeometry.flat_index",
                            geometry.flat_index(&signed),
                            flat(&target, &shape),
                        );
                    }
                }
            }
            for components in [1, 3, 17] {
                let input = data(n * components, 5);
                let mut expected = vec![0.0; input.len()];
                for index in 0..n {
                    let coordinate = coords(index, &shape);
                    for c in 0..components {
                        for axis in 0..shape.len() {
                            let mut plus = coordinate.clone();
                            let mut minus = coordinate.clone();
                            plus[axis] =
                                normalize(coordinate[axis] as isize + 1, shape[axis], boundary);
                            minus[axis] =
                                normalize(coordinate[axis] as isize - 1, shape[axis], boundary);
                            expected[index * components + c] += (input
                                [flat(&plus, &shape) * components + c]
                                + input[flat(&minus, &shape) * components + c]
                                - 2.0 * input[index * components + c])
                                / (spacing[axis] * spacing[axis]);
                        }
                    }
                }
                let mut output = vec![91.0; input.len() + 2];
                geometry
                    .laplacian(&input, components, &mut output[1..input.len() + 1])
                    .unwrap();
                report.close(
                    &format!("SquareLatticeGeometry.laplacian {boundary:?}"),
                    &output[1..input.len() + 1],
                    &expected,
                    ATOL,
                    RTOL,
                );
                assert_eq!(output[0], 91.0);
                assert_eq!(output[input.len() + 1], 91.0);
            }
            let values = data(n, 7);
            let lattice = SquareLattice::new(
                geometry.clone(),
                SquareLatticeInitMethod::Values {
                    values: values.clone(),
                },
            )
            .unwrap();
            let target: Vec<_> = shape.iter().map(|&n| (n / 2).max(1)).collect();
            let expected: Vec<_> = (0..target.iter().product())
                .map(|i| {
                    let source: Vec<_> = coords(i, &target)
                        .iter()
                        .enumerate()
                        .map(|(axis, &c)| c * shape[axis] / target[axis])
                        .collect();
                    values[flat(&source, &shape)]
                })
                .collect();
            report.close(
                "SquareLatticeAdvanced.downsample",
                lattice.downsample(&target).data(),
                &expected,
                0.0,
                0.0,
            );
            let mut changed = lattice.clone();
            let mut reference = values.clone();
            changed.set_flat(-1, 2.25);
            reference[normalize(-1, n, boundary)] = 2.25;
            let first = vec![0; shape.len()];
            changed.set(&first, -0.5);
            reference[0] = -0.5;
            report.close(
                "SquareLattice access/mutation",
                changed.data(),
                &reference,
                0.0,
                0.0,
            );
            let decoded: SquareLattice<f64> =
                serde_json::from_str(&serde_json::to_string(&changed).unwrap()).unwrap();
            report.close("SquareLattice serde", decoded.data(), &reference, 0.0, 0.0);
            changed.fill(1.75);
            report.close(
                "SquareLattice.fill",
                changed.data(),
                &vec![1.75; n],
                0.0,
                0.0,
            );
            let seeded = SquareLattice::new(
                geometry.clone(),
                SquareLatticeInitMethod::SeededCenter { val: 3.0 },
            )
            .unwrap();
            let mut seed = vec![0.0; n];
            seed[flat(&shape.iter().map(|n| n / 2).collect::<Vec<_>>(), &shape)] = 3.0;
            report.close("SquareLattice.SeededCenter", seeded.data(), &seed, 0.0, 0.0);
            for method in [
                SquareLatticeInitMethod::Empty,
                SquareLatticeInitMethod::Uniform { val: 0.0 },
            ] {
                report.close(
                    "SquareLattice.Empty/Uniform",
                    SquareLattice::new(geometry.clone(), method).unwrap().data(),
                    &vec![0.0; n],
                    0.0,
                    0.0,
                );
            }
        }
    }
    report.finish();
}

/// Slow repeated wall crossings, with no modulo/folding optimization.
pub(super) fn wall(mut r: f64, mut v: f64, lo: f64, hi: f64, kind: usize) -> (f64, f64) {
    match kind {
        0 => {
            while r < lo {
                r += hi - lo;
            }
            while r >= hi {
                r -= hi - lo;
            }
        }
        1 => {
            while r < lo || r > hi {
                if r < lo {
                    r = 2.0 * lo - r;
                    v = -v;
                }
                if r > hi {
                    r = 2.0 * hi - r;
                    v = -v;
                }
            }
        }
        _ => {
            r = r.max(lo).min(hi);
        }
    }
    (r, v)
}

#[test]
fn continuous_boundaries_single_batch_masks_and_particle_adapter() {
    let mut report = Report::default();
    for dim in [1, 2, 3, 7] {
        let lo = vec![-1.0; dim];
        let hi = vec![2.0; dim];
        let boundaries: Vec<Box<dyn ParticleBoundary>> = vec![
            Box::new(PeriodicBox::new(&lo, &hi).unwrap()),
            Box::new(ReflectBox::new(&lo, &hi).unwrap()),
            Box::new(ClampBox::new(&lo, &hi).unwrap()),
        ];
        for (kind, boundary) in boundaries.iter().enumerate() {
            for n in [1, 37, 2351] {
                let r: Vec<_> = (0..n * dim).map(|i| i as f64 % 61.0 - 30.125).collect();
                let v = data(n * dim, 11);
                let expected: Vec<_> = r
                    .iter()
                    .zip(&v)
                    .map(|(&r, &v)| wall(r, v, -1.0, 2.0, kind))
                    .collect();
                let er: Vec<_> = expected.iter().map(|x| x.0).collect();
                let ev: Vec<_> = expected.iter().map(|x| x.1).collect();
                let mut ar = r.clone();
                let mut av = v.clone();
                boundary
                    .apply_positions_velocities(&mut ar, &mut av)
                    .unwrap();
                report.close(
                    &format!("ContinuousBoundary.batch positions kind={kind}"),
                    &ar,
                    &er,
                    ATOL,
                    RTOL,
                );
                report.close(
                    &format!("ContinuousBoundary.batch velocities kind={kind}"),
                    &av,
                    &ev,
                    ATOL,
                    RTOL,
                );
                ar.copy_from_slice(&r);
                boundary.apply_positions(&mut ar).unwrap();
                report.close("ContinuousBoundary.apply_positions", &ar, &er, ATOL, RTOL);
                for row in [0, n - 1] {
                    let span = row * dim..(row + 1) * dim;
                    let mut ar = r[span.clone()].to_vec();
                    let mut av = v[span.clone()].to_vec();
                    boundary.apply_position_velocity(&mut ar, &mut av).unwrap();
                    report.close(
                        "ContinuousBoundary.single r/v",
                        &ar.iter().chain(&av).copied().collect::<Vec<_>>(),
                        &er[span.clone()]
                            .iter()
                            .chain(&ev[span.clone()])
                            .copied()
                            .collect::<Vec<_>>(),
                        ATOL,
                        RTOL,
                    );
                    ar.copy_from_slice(&r[span.clone()]);
                    let mut mask = vec![77; dim];
                    boundary
                        .apply_position_with_velocity_flip_mask(&mut ar, &mut mask)
                        .unwrap();
                    report.close(
                        "ContinuousBoundary.flip_mask positions",
                        &ar,
                        &er[span.clone()],
                        ATOL,
                        RTOL,
                    );
                    for axis in 0..dim {
                        assert_eq!(
                            mask[axis],
                            u8::from(
                                v[row * dim + axis].to_bits() != ev[row * dim + axis].to_bits()
                            )
                        );
                    }
                    boundary.apply_position(&mut ar).unwrap();
                    report.close(
                        "ContinuousBoundary.idempotent position",
                        &ar,
                        &er[span.clone()],
                        ATOL,
                        RTOL,
                    );
                }
                for backend in BACKENDS {
                    let mut particles = create_template(dim, n).unwrap();
                    for label in [ATTR_R, ATTR_V] {
                        particles
                            .attribute_mut::<f64>(label)
                            .unwrap()
                            .set_backend(backend);
                    }
                    for row in 0..n {
                        particles
                            .set_attribute_vector(ATTR_R, row, &r[row * dim..(row + 1) * dim])
                            .unwrap();
                        particles
                            .set_attribute_vector(ATTR_V, row, &v[row * dim..(row + 1) * dim])
                            .unwrap();
                        if row % 5 == 0 {
                            set_alive(&mut particles, row, false).unwrap();
                        }
                        if row % 7 == 0 {
                            set_rigid(&mut particles, row, true).unwrap();
                        }
                    }
                    boundary.apply_to_particles(&mut particles).unwrap();
                    let mut pr = er.clone();
                    let mut pv = ev.clone();
                    for row in 0..n {
                        if row % 5 == 0 || row % 7 == 0 {
                            pr[row * dim..(row + 1) * dim]
                                .copy_from_slice(&r[row * dim..(row + 1) * dim]);
                            pv[row * dim..(row + 1) * dim]
                                .copy_from_slice(&v[row * dim..(row + 1) * dim]);
                        }
                    }
                    report.close(
                        "ParticleBoundary.positions",
                        &particles
                            .attribute::<f64>(ATTR_R)
                            .unwrap()
                            .values()
                            .collect::<Vec<_>>(),
                        &pr,
                        ATOL,
                        RTOL,
                    );
                    report.close(
                        "ParticleBoundary.velocities",
                        &particles
                            .attribute::<f64>(ATTR_V)
                            .unwrap()
                            .values()
                            .collect::<Vec<_>>(),
                        &pv,
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
fn occupied_cells_and_particle_neighbor_queries_against_all_pairs() {
    let mut report = Report::default();
    for dim in [1, 2, 3, 5] {
        for backend in BACKENDS {
            let n = 67;
            let cutoff = 0.75;
            let mut p = data(n * dim, 7);
            let mut particles = create_template(dim, n).unwrap();
            particles
                .attribute_mut::<f64>(ATTR_R)
                .unwrap()
                .set_backend(backend);
            let mut raw = NeighborList::new(&vec![-3.0; dim], &vec![3.0; dim], cutoff).unwrap();
            let mut model =
                ParticleNeighborList::from_bounds(&vec![-3.0; dim], &vec![3.0; dim], cutoff)
                    .unwrap();
            for step in 0..3 {
                p[step * dim] += 0.125; // rebuild after motion; include coincident and exact-cutoff pairs
                for axis in 0..dim {
                    p[dim + axis] = p[axis];
                    p[2 * dim + axis] = p[axis];
                }
                p[2 * dim] += cutoff;
                for i in 0..n {
                    particles
                        .set_attribute_vector(ATTR_R, i, &p[i * dim..(i + 1) * dim])
                        .unwrap();
                    set_alive(&mut particles, i, i % 5 != step).unwrap();
                }
                raw.rebuild(&p, n).unwrap();
                model.rebuild(&particles).unwrap();
                let cells: Vec<Vec<_>> = p
                    .chunks(dim)
                    .map(|row| {
                        row.iter()
                            .map(|&x| (((x + 3.0) / cutoff).floor() as isize).clamp(0, 7))
                            .collect()
                    })
                    .collect();
                let mut candidates = BTreeSet::new();
                for i in 0..n {
                    for j in i + 1..n {
                        if (0..dim).all(|a| (cells[i][a] - cells[j][a]).abs() <= 1) {
                            candidates.insert((i, j));
                        }
                    }
                }
                let actual = raw.collect_pair_candidates();
                assert_eq!(actual.len(), actual.iter().collect::<BTreeSet<_>>().len());
                report.exact(
                    "NeighborList candidates",
                    actual.into_iter().collect::<BTreeSet<_>>(),
                    candidates.clone(),
                );
                let mut visited = BTreeSet::new();
                raw.for_each_pair_candidate(|i, j| {
                    assert!(visited.insert((i, j)));
                });
                report.exact("NeighborList callback", visited, candidates);
                for selection in [ParticleSelection::All, ParticleSelection::AliveOnly] {
                    let mut expected = BTreeSet::new();
                    for i in 0..n {
                        for j in i + 1..n {
                            if selection == ParticleSelection::AliveOnly
                                && (i % 5 == step || j % 5 == step)
                            {
                                continue;
                            }
                            let d2 = (0..dim)
                                .map(|a| (p[i * dim + a] - p[j * dim + a]).powi(2))
                                .sum::<f64>();
                            if d2 > 0.0 && d2 < cutoff * cutoff {
                                expected.insert((i, j));
                            }
                        }
                    }
                    let mut out = vec![(999, 1000)];
                    model
                        .collect_pairs_into(&particles, selection, &mut out)
                        .unwrap();
                    report.exact(
                        "ParticleNeighborList.collect_pairs_into",
                        out.into_iter().collect::<BTreeSet<_>>(),
                        expected.clone(),
                    );
                    report.exact(
                        "ParticleNeighborList.collect_pairs",
                        model
                            .collect_pairs(&particles, selection)
                            .unwrap()
                            .into_iter()
                            .collect::<BTreeSet<_>>(),
                        expected.clone(),
                    );
                    model
                        .rebuild_and_collect_into(&particles, selection, &mut Vec::new())
                        .unwrap();
                    report.exact(
                        "ParticleNeighborList.rebuild_and_collect",
                        model
                            .rebuild_and_collect(&particles, selection)
                            .unwrap()
                            .into_iter()
                            .collect::<BTreeSet<_>>(),
                        expected.clone(),
                    );
                    let mut visited = BTreeSet::new();
                    model
                        .for_each_pair(&particles, selection, |i, j| {
                            assert!(visited.insert((i, j)));
                        })
                        .unwrap();
                    report.exact("ParticleNeighborList.for_each_pair", visited, expected);
                }
            }
            raw.clear();
            assert!(raw.collect_pair_candidates().is_empty());
        }
    }
    report.finish();
}

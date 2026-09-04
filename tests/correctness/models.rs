use super::random::normal;
use super::support::*;
use physics_in_parallel::prelude::basic::*;
use physics_in_parallel::prelude::models::*;

fn state(
    dim: usize,
    n: usize,
    backend: Backend,
    r: &[f64],
    v: &[f64],
    a: &[f64],
    masked: bool,
) -> PhysObj {
    let mut particles = create_template(dim, n).unwrap();
    for label in [ATTR_R, ATTR_V, ATTR_A, ATTR_M, ATTR_M_INV] {
        particles
            .attribute_mut::<f64>(label)
            .unwrap()
            .set_backend(backend);
    }
    for i in 0..n {
        for (label, values) in [(ATTR_R, r), (ATTR_V, v), (ATTR_A, a)] {
            particles
                .set_attribute_vector(label, i, &values[i * dim..(i + 1) * dim])
                .unwrap();
        }
        set_mass(&mut particles, i, 0.75 + (i % 7) as f64 * 0.5).unwrap();
        if masked {
            set_alive(&mut particles, i, i % 5 != 0).unwrap();
            set_rigid(&mut particles, i, i % 7 == 0).unwrap();
        }
    }
    particles
}
fn values(p: &PhysObj, label: &str) -> Vec<f64> {
    p.attribute::<f64>(label).unwrap().values().collect()
}

#[test]
fn integrators_and_observers_match_particle_by_particle_reference() {
    let mut report = Report::default();
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for backend in BACKENDS {
                for dim in [1, 3, 7] {
                    for n in [1, 43, 16403] {
                        for masked in [false, true] {
                            let r = data(n * dim, 3);
                            let v = data(n * dim, 7);
                            let a = data(n * dim, 11);
                            let base = state(dim, n, backend, &r, &v, &a, masked);
                            for semi in [false, true] {
                                let mut p = base.clone();
                                let mut er = r.clone();
                                let mut ev = v.clone();
                                let mut integrator: Box<dyn Integrator> = if semi {
                                    Box::new(SemiImplicitEuler)
                                } else {
                                    Box::new(ExplicitEuler)
                                };
                                for dt in [0.125, 0.003, 0.017, 0.125, 0.001] {
                                    for i in 0..n {
                                        if masked && (i % 5 == 0 || i % 7 == 0) {
                                            continue;
                                        }
                                        for c in 0..dim {
                                            let j = i * dim + c;
                                            let old = ev[j];
                                            ev[j] += a[j] * dt;
                                            er[j] += dt * if semi { ev[j] } else { old };
                                        }
                                    }
                                    integrator.apply(&mut p, dt).unwrap();
                                    report.close(
                                        if semi {
                                            "SemiImplicitEuler.position"
                                        } else {
                                            "ExplicitEuler.position"
                                        },
                                        &values(&p, ATTR_R),
                                        &er,
                                        ATOL,
                                        RTOL,
                                    );
                                    report.close(
                                        "Integrator.velocity",
                                        &values(&p, ATTR_V),
                                        &ev,
                                        ATOL,
                                        RTOL,
                                    );
                                    report.close(
                                        "Integrator.acceleration unchanged",
                                        &values(&p, ATTR_A),
                                        &a,
                                        0.0,
                                        0.0,
                                    );
                                }
                                for selection in
                                    [ParticleSelection::All, ParticleSelection::AliveOnly]
                                {
                                    let mut energy = 0.0;
                                    let mut count = 0;
                                    for i in 0..n {
                                        if masked
                                            && selection == ParticleSelection::AliveOnly
                                            && i % 5 == 0
                                        {
                                            continue;
                                        }
                                        count += 1;
                                        let mass = 0.75 + (i % 7) as f64 * 0.5;
                                        for c in 0..dim {
                                            energy +=
                                                0.5 * mass * ev[i * dim + c] * ev[i * dim + c];
                                        }
                                    }
                                    let temperature = if count == 0 {
                                        0.0
                                    } else {
                                        2.0 * energy / (count * dim) as f64
                                    };
                                    let summary = kinetic_summary(&p, selection).unwrap();
                                    report.exact(
                                        "kinetic_summary.count",
                                        summary.particle_count,
                                        count,
                                    );
                                    report.close(
                                        "kinetic_summary.energy/temperature",
                                        &[summary.energy, summary.temperature],
                                        &[energy, temperature],
                                        ATOL,
                                        RTOL,
                                    );
                                    report.close(
                                        "Observer.energy/temperature",
                                        &[
                                            KineticEnergyObserver::new(selection)
                                                .observe(&p)
                                                .unwrap(),
                                            TemperatureObserver::new(selection)
                                                .observe(&p)
                                                .unwrap(),
                                        ],
                                        &[energy, temperature],
                                        ATOL,
                                        RTOL,
                                    );
                                }
                            }
                            for i in [0, n - 1] {
                                report.close(
                                    "set_mass/inverse_mass",
                                    &[p_mass(&base, ATTR_M, i), p_mass(&base, ATTR_M_INV, i)],
                                    &[
                                        0.75 + (i % 7) as f64 * 0.5,
                                        1.0 / (0.75 + (i % 7) as f64 * 0.5),
                                    ],
                                    0.0,
                                    0.0,
                                );
                                assert_eq!(is_alive(&base, i).unwrap(), !masked || i % 5 != 0);
                                assert_eq!(is_rigid(&base, i).unwrap(), masked && i % 7 == 0);
                            }
                        }
                    }
                }
            }
        });
    }
    report.finish();
}
fn p_mass(p: &PhysObj, label: &str, i: usize) -> f64 {
    p.attribute::<f64>(label).unwrap().get(i, 0).unwrap()
}

#[test]
fn spring_and_power_law_networks_match_naive_pair_sums() {
    let mut report = Report::default();
    for backend in BACKENDS {
        for dim in [1, 2, 3, 7] {
            for masked in [false, true] {
                for selection in [ParticleSelection::All, ParticleSelection::AliveOnly] {
                    let n = 31;
                    let mut r = data(n * dim, 3);
                    let v = data(n * dim, 7);
                    let a = data(n * dim, 11);
                    for c in 0..dim {
                        r[dim + c] = r[c];
                    } // coincident pair contributes zero
                    let base = state(dim, n, backend, &r, &v, &a, masked);
                    for power in [false, true] {
                        let mut springs = SpringNetwork::new();
                        let mut powers = PowerLawNetwork::new();
                        let mut laws = Vec::new();
                        for i in 0..n {
                            for j in i + 1..n {
                                let k = if (i + j) % 3 == 0 { -0.75 } else { 1.25 };
                                let parameter = if power {
                                    -1.5 + (i % 3) as f64
                                } else {
                                    0.5 + (i % 3) as f64
                                };
                                let range = if (i + j) % 2 == 0 {
                                    Some((0.5, 4.0))
                                } else {
                                    None
                                };
                                springs
                                    .insert((j, i), Spring::new(k, parameter.abs(), range).unwrap())
                                    .unwrap();
                                powers
                                    .insert(
                                        (j, i),
                                        PowerLawDecay::new(k, parameter, range).unwrap(),
                                    )
                                    .unwrap();
                                laws.push((i, j, k, parameter, range));
                            }
                        }
                        let mut expected = a.clone();
                        let mut p = base.clone();
                        for _ in 0..3 {
                            // Plain pair traversal uses force magnitude times unit direction.
                            // The production path folds the distance into a single scale.
                            for &(i, j, k, parameter, range) in &laws {
                                if masked
                                    && selection == ParticleSelection::AliveOnly
                                    && (i % 5 == 0 || j % 5 == 0)
                                {
                                    continue;
                                }
                                let delta: Vec<_> =
                                    (0..dim).map(|c| r[j * dim + c] - r[i * dim + c]).collect();
                                let squared = delta.iter().map(|x| x * x).sum::<f64>();
                                if squared <= f64::EPSILON {
                                    continue;
                                }
                                let distance = squared.sqrt();
                                if range.is_some_and(|(lo, hi)| distance < lo || distance > hi) {
                                    continue;
                                }
                                let magnitude = if power {
                                    -k * distance.powf(parameter)
                                } else {
                                    k * (distance - parameter.abs())
                                };
                                for c in 0..dim {
                                    let force = magnitude * (delta[c] / distance);
                                    if !masked || i % 7 != 0 {
                                        expected[i * dim + c] +=
                                            force / (0.75 + (i % 7) as f64 * 0.5);
                                    }
                                    if !masked || j % 7 != 0 {
                                        expected[j * dim + c] -=
                                            force / (0.75 + (j % 7) as f64 * 0.5);
                                    }
                                }
                            }
                            if power {
                                powers.apply(&mut p, selection).unwrap();
                            } else {
                                springs.apply(&mut p, selection).unwrap();
                            }
                            report.close(
                                if power {
                                    "PowerLawNetwork.apply"
                                } else {
                                    "SpringNetwork.apply"
                                },
                                &values(&p, ATTR_A),
                                &expected,
                                ATOL,
                                RTOL,
                            );
                        }
                        report.close(
                            "Networks.positions unchanged",
                            &values(&p, ATTR_R),
                            &r,
                            0.0,
                            0.0,
                        );
                        // Public payload and persistence routes retain the explicitly supplied laws.
                        let json = serde_json::to_string(&springs).unwrap();
                        let restored: SpringNetwork = serde_json::from_str(&json).unwrap();
                        assert_eq!(restored.len(), laws.len());
                        for &(i, j, k, param, range) in &laws {
                            let s = restored.get((i, j)).unwrap();
                            assert_eq!(
                                (s.spring_constant(), s.rest_length(), s.cutoff()),
                                (k, param.abs(), range)
                            );
                        }
                        let restored: PowerLawNetwork =
                            serde_json::from_str(&serde_json::to_string(&powers).unwrap()).unwrap();
                        for &(i, j, k, param, range) in &laws {
                            let p = restored.get((i, j)).unwrap();
                            assert_eq!((p.strength(), p.exponent(), p.range()), (k, param, range));
                        }
                    }
                }
            }
        }
    }
    let mut all = PowerLawNetwork::new();
    all.insert_all_to_all(7, PowerLawDecay::new(1.0, -2.0, None).unwrap())
        .unwrap();
    report.exact("PowerLawNetwork.all_to_all", all.len(), 21);
    report.finish();
}

#[test]
fn langevin_thermostat_matches_scalar_ornstein_uhlenbeck_steps() {
    let mut report = Report::default();
    let seed = 19;
    let rng = ResolvedRng::new(seed, RngMethod::IndexedSplitMix64);
    for backend in BACKENDS {
        for dim in [1, 3, 7] {
            for n in [1, 43, 2351] {
                for selection in [ParticleSelection::All, ParticleSelection::AliveOnly] {
                    for (tau, gamma) in [(0.0, 0.75), (1.75, 0.0), (1.75, 0.75)] {
                        let r = data(n * dim, 3);
                        let v = data(n * dim, 7);
                        let a = data(n * dim, 11);
                        let mut p = state(dim, n, backend, &r, &v, &a, true);
                        let mut expected = v.clone();
                        let mut thermostat =
                            LangevinThermostat::from_state(tau, gamma, rng, 7, selection).unwrap();
                        for step in 7..12 {
                            let dt = 0.013;
                            let decay = (-gamma * dt).exp();
                            for i in 0..n {
                                if i % 7 == 0
                                    || selection == ParticleSelection::AliveOnly && i % 5 == 0
                                {
                                    continue;
                                }
                                let mass = 0.75 + (i % 7) as f64 * 0.5;
                                let sigma = (tau / mass * (-(-2.0 * gamma * dt).exp_m1())).sqrt();
                                for c in 0..dim {
                                    expected[i * dim + c] = decay * expected[i * dim + c]
                                        + sigma
                                            * normal(
                                                seed,
                                                step,
                                                0xc7c7d2529b536071,
                                                i as u64,
                                                c as u64,
                                            );
                                }
                            }
                            thermostat.apply(&mut p, dt).unwrap();
                            report.close(
                                "LangevinThermostat.velocity",
                                &values(&p, ATTR_V),
                                &expected,
                                ATOL,
                                RTOL,
                            );
                            report.exact(
                                "LangevinThermostat.step",
                                thermostat.step_counter(),
                                step + 1,
                            );
                        }
                        report.close(
                            "LangevinThermostat.positions unchanged",
                            &values(&p, ATTR_R),
                            &r,
                            0.0,
                            0.0,
                        );
                    }
                }
            }
        }
    }
    report.finish();
}

#[test]
fn coupled_spring_trajectory_matches_naive_simulator() {
    let mut report = Report::default();
    let n = 17;
    let dim = 3;
    let dt = 0.002;
    let r = data(n * dim, 3);
    let v = data(n * dim, 7);
    let masses: Vec<_> = (0..n).map(|i| 0.75 + (i % 7) as f64 * 0.5).collect();
    for backend in BACKENDS {
        let mut p = state(dim, n, backend, &r, &v, &vec![0.0; n * dim], false);
        let mut network = SpringNetwork::new();
        for i in 0..n - 1 {
            network
                .insert((i, i + 1), Spring::new(1.25, 0.75, None).unwrap())
                .unwrap();
        }
        let mut er = r.clone();
        let mut ev = v.clone();
        for _ in 0..100 {
            let mut forces = vec![0.0; n * dim];
            for i in 0..n - 1 {
                let distance = (0..dim)
                    .map(|c| (er[(i + 1) * dim + c] - er[i * dim + c]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                for c in 0..dim {
                    let force =
                        1.25 * (distance - 0.75) * (er[(i + 1) * dim + c] - er[i * dim + c])
                            / distance;
                    forces[i * dim + c] += force;
                    forces[(i + 1) * dim + c] -= force;
                }
            }
            for i in 0..n {
                for c in 0..dim {
                    ev[i * dim + c] += dt * forces[i * dim + c] / masses[i];
                    er[i * dim + c] += dt * ev[i * dim + c];
                }
            }
            p.attribute_mut::<f64>(ATTR_A).unwrap().fill(0.0);
            network.apply(&mut p, ParticleSelection::All).unwrap();
            SemiImplicitEuler.apply(&mut p, dt).unwrap();
        }
        report.close(
            "coupled 100-step trajectory.position",
            &values(&p, ATTR_R),
            &er,
            ATOL,
            RTOL,
        );
        report.close(
            "coupled 100-step trajectory.velocity",
            &values(&p, ATTR_V),
            &ev,
            ATOL,
            RTOL,
        );
    }
    report.finish();
}

/*!
A **particle model benchmark** comparing PiP's SoA power-law network against a naive AoS baseline.

This example measures the total simulation lifecycle:
- **Initiation**: Allocation of particle storage and population of interaction networks.
- **Evolution**: Numerical time-stepping including power-law acceleration and integration.

Power-law networks are constructed as all-to-all particle models here: every
unordered particle pair `(i, j)` with `i < j` receives the same payload.

For a complete PiP model usage example, read `run_pip_version`. It shows the
full procedure: create canonical particle state, write initial positions, build
an all-to-all power-law network, apply power-law acceleration, and integrate the
state.

Metrics are displayed in a comparative chart, including numerical validation between implementations.
*/

use physics_in_parallel::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

const DEFAULT_NUM_PARTICLES: usize = 1_000;
const DEFAULT_STEPS: usize = 100;

#[derive(Clone, Copy, Debug)]
struct NaiveParticle {
    r: [f64; 3],
    v: [f64; 3],
    a: [f64; 3],
    m_inv: f64,
}

struct NaiveBenchmark {
    particles: Vec<NaiveParticle>,
    init_ms: f64,
    evol_ms_per_step: f64,
}

struct PipBenchmark {
    objects: PhysObj,
    init_ms: f64,
    evol_ms_per_step: f64,
}

/// Naive AoS evolution step.
/// Positive `k` is repulsive, matching `PowerLawNetwork::apply_power_law_acceleration`.
fn naive_step(
    particles: &mut [NaiveParticle],
    interactions: &[(usize, usize, PowerLawDecay)],
    dt: f64,
) {
    particles.par_iter_mut().for_each(|p| p.a = [0.0; 3]);

    for (i, j, law) in interactions {
        let p1 = particles[*i];
        let p2 = particles[*j];
        let dx = [p1.r[0] - p2.r[0], p1.r[1] - p2.r[1], p1.r[2] - p2.r[2]];
        let d2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
        if !d2.is_finite() || d2 <= f64::EPSILON {
            continue;
        }

        let dist = d2.sqrt();
        if let Some((min, max)) = law.range
            && (dist < min || dist > max)
        {
            continue;
        }

        let scale = law.k * dist.powf(law.alpha - 1.0);
        let fx = dx[0] * scale;
        let fy = dx[1] * scale;
        let fz = dx[2] * scale;

        particles[*i].a[0] += fx * p1.m_inv;
        particles[*i].a[1] += fy * p1.m_inv;
        particles[*i].a[2] += fz * p1.m_inv;

        particles[*j].a[0] -= fx * p2.m_inv;
        particles[*j].a[1] -= fy * p2.m_inv;
        particles[*j].a[2] -= fz * p2.m_inv;
    }

    particles.par_iter_mut().for_each(|p| {
        p.v[0] += p.a[0] * dt;
        p.r[0] += p.v[0] * dt;
        p.v[1] += p.a[1] * dt;
        p.r[1] += p.v[1] * dt;
        p.v[2] += p.a[2] * dt;
        p.r[2] += p.v[2] * dt;
    });
}

fn all_to_all_pair_count(n: usize) -> usize {
    n.saturating_mul(n.saturating_sub(1)) / 2
}

fn build_all_to_all_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(all_to_all_pair_count(n));

    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }

    pairs
}

/// Full naive AoS baseline lifecycle: allocate state, add interactions, and evolve.
fn run_naive_version(
    n: usize,
    pairs: &[(usize, usize)],
    law: PowerLawDecay,
    dt: f64,
    steps: usize,
) -> NaiveBenchmark {
    let start_init = Instant::now();
    let mut particles = vec![
        NaiveParticle {
            r: [0.0; 3],
            v: [0.0; 3],
            a: [0.0; 3],
            m_inv: 1.0
        };
        n
    ];
    let mut interactions = Vec::with_capacity(pairs.len());
    for i in 0..n {
        particles[i].r = [i as f64 * 0.001, 0.0, 0.0];
    }
    for &(u, v) in pairs {
        interactions.push((u, v, law));
    }
    let init_ms = start_init.elapsed().as_secs_f64() * 1000.0;

    let start_evol = Instant::now();
    for _ in 0..steps {
        naive_step(&mut particles, &interactions, dt);
    }
    let evol_ms_per_step = start_evol.elapsed().as_secs_f64() * 1000.0 / steps as f64;

    NaiveBenchmark {
        particles,
        init_ms,
        evol_ms_per_step,
    }
}

/// Full PiP model lifecycle and usage demonstration.
///
/// This is the canonical example to copy from when building a power-law
/// particle model with PiP:
/// 1. Create canonical massive-particle state with `create_template`.
/// 2. Populate the `ATTR_R` position attribute.
/// 3. Build a capacity-aware all-to-all `PowerLawNetwork`.
/// 4. Add the payload to every unordered particle pair.
/// 5. For each step, clear acceleration, apply power-law acceleration, and integrate.
fn run_pip_version(n: usize, law: PowerLawDecay, dt: f64, steps: usize) -> PipBenchmark {
    let start_init = Instant::now();
    let mut objects = create_template(3, n).unwrap();
    let mut interactions = PowerLawNetwork::all_to_all_empty(n);

    {
        let r = objects.core.get_mut::<f64>(ATTR_R).unwrap();
        for i in 0..n {
            r.set_vec(i as isize, &[i as f64 * 0.001, 0.0, 0.0]);
        }
    }

    interactions.add_all_to_all_payload(n, law).unwrap();
    let init_ms = start_init.elapsed().as_secs_f64() * 1000.0;

    let start_evol = Instant::now();
    let mut integrator = SemiImplicitEuler;
    for _ in 0..steps {
        objects.core.get_mut::<f64>(ATTR_A).unwrap().fill(0.0);
        interactions
            .apply_power_law_acceleration(&mut objects, ParticleSelection::All)
            .unwrap();
        integrator.apply(&mut objects, dt).unwrap();
    }
    let evol_ms_per_step = start_evol.elapsed().as_secs_f64() * 1000.0 / steps as f64;

    PipBenchmark {
        objects,
        init_ms,
        evol_ms_per_step,
    }
}

fn run_comparison(n: usize, steps: usize) {
    let dt = 0.001;
    let pairs = build_all_to_all_pairs(n);
    let law = PowerLawDecay::new(0.0001, -2.0, Some((0.01, 5.0))).unwrap();

    let naive = run_naive_version(n, &pairs, law, dt, steps);
    let pip = run_pip_version(n, law, dt, steps);

    let mut max_diff: f64 = 0.0;
    let pip_r = pip.objects.core.get::<f64>(ATTR_R).unwrap();
    for i in 0..n {
        let p_r = pip_r.get_vec(i as isize);
        for axis in 0..3 {
            max_diff = max_diff.max((naive.particles[i].r[axis] - p_r[axis]).abs());
        }
    }
    let validation_passed = max_diff < 1e-10;

    print_chart(
        n,
        pairs.len(),
        naive.init_ms,
        pip.init_ms,
        naive.evol_ms_per_step,
        pip.evol_ms_per_step,
        max_diff,
        validation_passed,
    );
}

fn print_chart(
    n: usize,
    m: usize,
    n_init: f64,
    p_init: f64,
    n_evol: f64,
    p_evol: f64,
    diff: f64,
    passed: bool,
) {
    println!("\n================================================================================");
    println!(
        " PIP BENCHMARK: Power-Law Network Evolution (N={}, M={})",
        n, m
    );
    println!("================================================================================");
    println!(
        "| {:<24} | {:>15} | {:>15} | {:>10} |",
        "Metric", "Naive (AoS)", "PiP (SoA)", "Speedup"
    );
    println!("|{:-<26}|{:-<17}|{:-<17}|{:-<12}|", "", "", "", "");

    println!(
        "| {:<24} | {:>12.3} ms | {:>12.3} ms | {:>9.2}x |",
        "Initiation Time",
        n_init,
        p_init,
        n_init / p_init
    );

    println!(
        "| {:<24} | {:>9.3} ms/step | {:>9.3} ms/step | {:>9.2}x |",
        "Evolution Time",
        n_evol,
        p_evol,
        n_evol / p_evol
    );

    let n_tp = (n as f64 / n_evol) * 1000.0;
    let p_tp = (n as f64 / p_evol) * 1000.0;
    println!(
        "| {:<24} | {:>11.2}k p/s | {:>11.2}k p/s | {:>9.2}x |",
        "Throughput",
        n_tp / 1000.0,
        p_tp / 1000.0,
        p_tp / n_tp
    );

    println!("|{:-<26}|{:-<17}|{:-<17}|{:-<12}|", "", "", "", "");
    println!(
        "| {:<24} | {:<47} |",
        "Numerical Validation",
        if passed {
            format!("PASSED (Max Diff: {:.2e})", diff)
        } else {
            "FAILED".to_string()
        }
    );
    println!("================================================================================");
}

fn main() {
    let n = parse_arg(1).unwrap_or(DEFAULT_NUM_PARTICLES);
    let steps = parse_arg(2).unwrap_or(DEFAULT_STEPS);

    println!("Starting benchmark...");
    println!("particles: {n}");
    println!("pairs: {}", all_to_all_pair_count(n));
    println!("steps: {steps}");

    run_comparison(n, steps);
}

fn parse_arg(position: usize) -> Option<usize> {
    std::env::args().nth(position)?.parse().ok()
}

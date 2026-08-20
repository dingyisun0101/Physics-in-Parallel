/*!
A **particle model benchmark** comparing the high-performance SoA backend against a naive AoS baseline.

This example measures the total simulation lifecycle:
- **Initiation**: Allocation of particle storage and population of interaction networks.
- **Evolution**: Numerical time-stepping including force accumulation and integration.

For a complete PiP model usage example, read `run_pip_version`. It shows the
full procedure: create canonical particle state, write initial positions, build
a spring network, apply spring acceleration, and integrate the state.

Metrics are displayed in a comparative chart, including numerical validation between implementations.
*/

use physics_in_parallel::prelude::models::*;
use rayon::prelude::*;
use std::time::Instant;

const DEFAULT_NUM_PARTICLES: usize = 10_000;
const DEFAULT_NUM_SPRINGS: usize = 4_000;
const DEFAULT_STEPS: usize = 1_000;

//===================================================================
// -------------------------- Naive Baseline ------------------------
//===================================================================

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
/// Uses Rayon for integration but serial loops for force accumulation.
fn naive_step(particles: &mut [NaiveParticle], springs: &[(usize, usize, Spring)], dt: f64) {
    // 1. Reset acceleration (Rayon)
    particles.par_iter_mut().for_each(|p| p.a = [0.0; 3]);

    // 2. Accumulate forces (Serial)
    for (i, j, law) in springs {
        let p1 = particles[*i];
        let p2 = particles[*j];
        let dx = [p1.r[0] - p2.r[0], p1.r[1] - p2.r[1], p1.r[2] - p2.r[2]];
        let d2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
        let dist = d2.sqrt();

        let max_dist = law.cutoff.map(|(_, max)| max).unwrap_or(f64::INFINITY);
        if dist > 0.0 && dist < max_dist {
            let f_mag = -law.k * (dist - law.l_0);
            let fx = (f_mag * dx[0]) / dist;
            let fy = (f_mag * dx[1]) / dist;
            let fz = (f_mag * dx[2]) / dist;

            particles[*i].a[0] += fx * p1.m_inv;
            particles[*i].a[1] += fy * p1.m_inv;
            particles[*i].a[2] += fz * p1.m_inv;

            particles[*j].a[0] -= fx * p2.m_inv;
            particles[*j].a[1] -= fy * p2.m_inv;
            particles[*j].a[2] -= fz * p2.m_inv;
        }
    }

    // 3. Integrate (Rayon)
    particles.par_iter_mut().for_each(|p| {
        p.v[0] += p.a[0] * dt;
        p.r[0] += p.v[0] * dt;
        p.v[1] += p.a[1] * dt;
        p.r[1] += p.v[1] * dt;
        p.v[2] += p.a[2] * dt;
        p.r[2] += p.v[2] * dt;
    });
}

fn build_spring_pairs(n: usize, m: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(m);

    for offset in 1..n {
        for u in 0..n {
            if pairs.len() == m {
                return pairs;
            }

            let v = (u + offset) % n;
            if u < v {
                pairs.push((u, v));
            }
        }
    }

    pairs
}

/// Full naive AoS baseline lifecycle: allocate state, add springs, and evolve.
fn run_naive_version(
    n: usize,
    spring_pairs: &[(usize, usize)],
    spring: Spring,
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
    let mut springs = Vec::with_capacity(spring_pairs.len());
    for (i, particle) in particles.iter_mut().enumerate().take(n) {
        particle.r = [i as f64 * 0.001, 0.0, 0.0];
    }
    for &(u, v) in spring_pairs {
        springs.push((u, v, spring));
    }
    let init_ms = start_init.elapsed().as_secs_f64() * 1000.0;

    let start_evol = Instant::now();
    for _ in 0..steps {
        naive_step(&mut particles, &springs, dt);
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
/// This is the canonical example to copy from when building a spring-network
/// particle model with PiP:
/// 1. Create canonical massive-particle state with `create_template`.
/// 2. Populate the `ATTR_R` position attribute.
/// 3. Build a capacity-aware `SpringNetwork`.
/// 4. Add spring payloads in batch.
/// 5. For each step, clear acceleration, apply Hooke acceleration, and integrate.
fn run_pip_version(
    n: usize,
    spring_pairs: &[(usize, usize)],
    spring: Spring,
    dt: f64,
    steps: usize,
) -> PipBenchmark {
    let start_init = Instant::now();
    let mut objects = create_template(3, n).unwrap();
    let mut springs = SpringNetwork::with_capacity(n, spring_pairs.len());

    {
        let r = objects.attribute_mut::<f64>(ATTR_R).unwrap();
        for i in 0..n {
            r.set_vector(i as isize, &[i as f64 * 0.001, 0.0, 0.0]);
        }
    }

    springs.add_springs_payload(spring_pairs, spring).unwrap();
    let init_ms = start_init.elapsed().as_secs_f64() * 1000.0;

    let start_evol = Instant::now();
    let mut integrator = SemiImplicitEuler;
    for _ in 0..steps {
        objects.attribute_mut::<f64>(ATTR_A).unwrap().fill(0.0);
        springs
            .apply_hooke_acceleration(&mut objects, ParticleSelection::All)
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

//===================================================================
// ------------------------- Benchmark Runner -----------------------
//===================================================================

fn run_comparison(n: usize, m: usize, steps: usize) {
    let dt = 0.001;
    let spring_pairs = build_spring_pairs(n, m);
    let spring = Spring::new(10.0, 1.0, Some((0.0, 5.0))).unwrap();

    let naive = run_naive_version(n, &spring_pairs, spring, dt, steps);
    let pip = run_pip_version(n, &spring_pairs, spring, dt, steps);

    // --- VALIDATION ---
    let mut max_diff: f64 = 0.0;
    let pip_r = pip.objects.attribute::<f64>(ATTR_R).unwrap();
    for (i, particle) in naive.particles.iter().enumerate().take(n) {
        let p_r = pip_r.vector(i as isize);
        for (axis, &pip_value) in p_r.iter().enumerate().take(3) {
            max_diff = max_diff.max((particle.r[axis] - pip_value).abs());
        }
    }
    let validation_passed = max_diff < 1e-10;

    // --- OUTPUT CHART ---
    print_chart(
        n,
        spring_pairs.len(),
        naive.init_ms,
        pip.init_ms,
        naive.evol_ms_per_step,
        pip.evol_ms_per_step,
        max_diff,
        validation_passed,
    );
}

#[allow(clippy::too_many_arguments)]
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
        " PIP BENCHMARK: Spring Network Evolution (N={}, M={})",
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
    let m = parse_arg(2).unwrap_or(DEFAULT_NUM_SPRINGS);
    let steps = parse_arg(3).unwrap_or(DEFAULT_STEPS);

    println!("Starting benchmark...");
    println!("particles: {n}");
    println!("springs: {m}");
    println!("steps: {steps}");

    run_comparison(n, m, steps);
}

fn parse_arg(position: usize) -> Option<usize> {
    std::env::args().nth(position)?.parse().ok()
}

use std::hint::black_box;
use std::time::{Duration, Instant};

use physics_in_parallel::math::tensor::rank_2::vector_list::{
    HaarVectors, VectorList, VectorListRand,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Normal};

const DEFAULT_DIM: usize = 3;
const DEFAULT_NUM_VECS: usize = 5_000_000;
const DEFAULT_REPEATS: usize = 3;
const SEED: u64 = 0xBADC_0FFE_EE11_2233;

fn main() {
    let dim = parse_arg(1).unwrap_or(DEFAULT_DIM);
    let num_vecs = parse_arg(2).unwrap_or(DEFAULT_NUM_VECS);
    let repeats = parse_arg(3).unwrap_or(DEFAULT_REPEATS);
    let rngs = parse_arg(4).unwrap_or_else(rayon::current_num_threads);

    println!("vector-list Haar generation benchmark");
    println!("dim: {dim}");
    println!("vectors: {num_vecs}");
    println!("scalar entries: {}", dim * num_vecs);
    println!(
        "storage: {}",
        bytes_human(dim * num_vecs * size_of::<f64>())
    );
    println!("repeats: {repeats}");
    println!("rayon threads available: {}", rayon::current_num_threads());
    println!("vector-list RNG shards: {rngs}");
    println!();

    let vector_list = benchmark_vector_list_haar(dim, num_vecs, repeats, rngs);
    let sequential = benchmark_sequential_haar(dim, num_vecs, repeats);

    println!(
        "{:<16} {:>14} {:>14} {:>14}",
        "method", "best", "average", "speedup"
    );
    println!("{}", "-".repeat(64));
    println!(
        "{:<16} {:>14} {:>14} {:>14}",
        "VectorList",
        millis(vector_list.best),
        millis(vector_list.average),
        "1.000"
    );
    println!(
        "{:<16} {:>14} {:>14} {:>14.3}",
        "Sequential",
        millis(sequential.best),
        millis(sequential.average),
        sequential.best.as_secs_f64() / vector_list.best.as_secs_f64()
    );
    println!();

    report_sum("VectorList", &vector_list.sum);
    report_sum("Sequential", &sequential.sum);
    println!();
    println!(
        "For independent Haar unit vectors, each component has mean zero. The component sums should therefore be small compared with the vector count, with random fluctuation on the order of sqrt(num_vecs / dim)."
    );
}

#[derive(Debug, Clone)]
struct Timing {
    best: Duration,
    average: Duration,
    sum: Vec<f64>,
}

fn benchmark_vector_list_haar(dim: usize, num_vecs: usize, repeats: usize, rngs: usize) -> Timing {
    let mut generator = HaarVectors::new(dim, num_vecs, Some(rngs));
    generator.refresh();

    let timings = (0..repeats)
        .map(|_| {
            let start = Instant::now();
            generator.refresh();
            start.elapsed()
        })
        .collect::<Vec<_>>();

    Timing {
        best: *timings.iter().min().expect("at least one repeat"),
        average: average(&timings),
        sum: black_box(vector_sum(&generator.vl)),
    }
}

fn benchmark_sequential_haar(dim: usize, num_vecs: usize, repeats: usize) -> Timing {
    let mut vectors = vec![0.0; dim * num_vecs];
    let mut rng = SmallRng::seed_from_u64(SEED);
    let normal = Normal::new(0.0, 1.0).expect("valid standard normal distribution");

    fill_sequential_haar(&mut vectors, dim, &normal, &mut rng);

    let timings = (0..repeats)
        .map(|_| {
            let start = Instant::now();
            fill_sequential_haar(&mut vectors, dim, &normal, &mut rng);
            start.elapsed()
        })
        .collect::<Vec<_>>();

    Timing {
        best: *timings.iter().min().expect("at least one repeat"),
        average: average(&timings),
        sum: black_box(sum_rows(&vectors, dim)),
    }
}

fn fill_sequential_haar<R>(vectors: &mut [f64], dim: usize, normal: &Normal<f64>, rng: &mut R)
where
    R: rand::Rng + ?Sized,
{
    for row in vectors.chunks_exact_mut(dim) {
        let mut norm_sqr = 0.0;
        for x in row.iter_mut() {
            *x = normal.sample(rng);
            norm_sqr += *x * *x;
        }

        let norm = norm_sqr.sqrt();
        if norm > 0.0 {
            for x in row {
                *x /= norm;
            }
        }
    }
}

fn vector_sum(vectors: &VectorList<f64>) -> Vec<f64> {
    let mut sum = vec![0.0; vectors.dim()];
    for i in 0..vectors.num_vecs() {
        for (axis, x) in vectors.get_vec(i as isize).iter().copied().enumerate() {
            sum[axis] += x;
        }
    }
    sum
}

fn sum_rows(data: &[f64], dim: usize) -> Vec<f64> {
    let mut sum = vec![0.0; dim];
    for row in data.chunks_exact(dim) {
        for (axis, x) in row.iter().copied().enumerate() {
            sum[axis] += x;
        }
    }
    sum
}

fn report_sum(label: &str, sum: &[f64]) {
    let l2 = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
    let max_abs = sum.iter().map(|x| x.abs()).fold(0.0, f64::max);
    println!(
        "{label:<10} component sum = {:?}, |sum|_2 = {:.6}, max |component| = {:.6}",
        sum, l2, max_abs
    );
}

fn average(times: &[Duration]) -> Duration {
    let total = times.iter().map(Duration::as_secs_f64).sum::<f64>();
    Duration::from_secs_f64(total / times.len() as f64)
}

fn millis(duration: Duration) -> String {
    format!("{:.3} ms", duration.as_secs_f64() * 1_000.0)
}

fn parse_arg(position: usize) -> Option<usize> {
    std::env::args().nth(position)?.parse().ok()
}

fn bytes_human(bytes: usize) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else {
        format!("{:.2} MiB", bytes / MIB)
    }
}

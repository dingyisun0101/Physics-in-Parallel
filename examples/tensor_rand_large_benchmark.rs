use std::hint::black_box;
use std::time::{Duration, Instant};

use physics_in_parallel::math::tensor::{RandType, RngKind, TensorRandFiller, TensorTrait, dense};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_chacha::{ChaCha8Rng, ChaCha12Rng, ChaCha20Rng};
use rand_distr::{Distribution, Uniform};
use rand_pcg::{Pcg64, Pcg64Mcg};

const DEFAULT_LEN: usize = 120_000_000;
const DEFAULT_REPEATS: usize = 3;
const SEED: u64 = 0x5EED_1234;
const DEFAULT_RNG_KINDS: &[RngKind] = &[RngKind::Pcg64, RngKind::Pcg64Mcg, RngKind::SmallRng];

fn main() {
    let len = parse_arg(1).unwrap_or(DEFAULT_LEN);
    let repeats = parse_arg(2).unwrap_or(DEFAULT_REPEATS);
    let rngs = parse_arg(3).unwrap_or_else(rayon::current_num_threads);
    let rng_kinds = parse_rng_kinds(4).unwrap_or_else(|| DEFAULT_RNG_KINDS.to_vec());
    let kind = RandType::Uniform {
        low: -1.0,
        high: 1.0,
    };

    println!("tensor random fill benchmark");
    println!("elements: {len}");
    println!("bytes per array: {}", bytes_human(len * size_of::<f64>()));
    println!("repeats: {repeats}");
    println!("rayon threads available: {}", rayon::current_num_threads());
    println!("rayon threads: {rngs}");
    println!(
        "rng kinds: {}",
        rng_kinds
            .iter()
            .map(|kind| kind.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!();

    println!(
        "{:<12} {:>14} {:>14} {:>14} {:>14} {:>12} {:>12}",
        "rng", "tensor best", "tensor avg", "seq best", "seq avg", "best x", "avg x"
    );
    println!("{}", "-".repeat(100));

    for rng_kind in rng_kinds {
        let tensor = benchmark_tensor_fill(len, repeats, kind, rngs, rng_kind);
        let sequential = benchmark_sequential_vec_fill(len, repeats, rng_kind);
        report(rng_kind, &tensor, &sequential);
    }
}

#[derive(Debug, Clone, Copy)]
struct Timing {
    best: Duration,
    average: Duration,
    checksum: f64,
}

fn benchmark_tensor_fill(
    len: usize,
    repeats: usize,
    kind: RandType,
    rngs: usize,
    rng_kind: RngKind,
) -> Timing {
    let mut tensor = dense::Tensor::<f64>::empty(&[len]);
    let mut filler = TensorRandFiller::new(kind, Some(SEED), Some(rng_kind), Some(rngs));

    filler.refresh(&mut tensor);

    let timings = (0..repeats)
        .map(|_| {
            let start = Instant::now();
            filler.refresh(&mut tensor);
            start.elapsed()
        })
        .collect::<Vec<_>>();

    Timing {
        best: *timings.iter().min().expect("at least one repeat"),
        average: average(&timings),
        checksum: black_box(tensor.get_sum()),
    }
}

fn benchmark_sequential_vec_fill(len: usize, repeats: usize, rng_kind: RngKind) -> Timing {
    match rng_kind {
        RngKind::SmallRng => {
            benchmark_sequential_vec_fill_with_rng(len, repeats, SmallRng::seed_from_u64(SEED))
        }
        RngKind::Pcg64Mcg => {
            benchmark_sequential_vec_fill_with_rng(len, repeats, Pcg64Mcg::seed_from_u64(SEED))
        }
        RngKind::Pcg64 => {
            benchmark_sequential_vec_fill_with_rng(len, repeats, Pcg64::seed_from_u64(SEED))
        }
        RngKind::ChaCha8 => {
            benchmark_sequential_vec_fill_with_rng(len, repeats, ChaCha8Rng::seed_from_u64(SEED))
        }
        RngKind::ChaCha12 => {
            benchmark_sequential_vec_fill_with_rng(len, repeats, ChaCha12Rng::seed_from_u64(SEED))
        }
        RngKind::ChaCha20 => {
            benchmark_sequential_vec_fill_with_rng(len, repeats, ChaCha20Rng::seed_from_u64(SEED))
        }
    }
}

fn benchmark_sequential_vec_fill_with_rng<R>(len: usize, repeats: usize, mut rng: R) -> Timing
where
    R: rand::Rng,
{
    let dist = Uniform::new(-1.0, 1.0).expect("valid uniform bounds");
    let mut data = vec![0.0; len];

    fill_sequential_with_rng(&mut data, &dist, &mut rng);

    let timings = (0..repeats)
        .map(|_| {
            let start = Instant::now();
            fill_sequential_with_rng(&mut data, &dist, &mut rng);
            start.elapsed()
        })
        .collect::<Vec<_>>();

    Timing {
        best: *timings.iter().min().expect("at least one repeat"),
        average: average(&timings),
        checksum: checksum(&data),
    }
}

fn fill_sequential_with_rng<R>(data: &mut [f64], dist: &Uniform<f64>, rng: &mut R)
where
    R: rand::Rng + ?Sized,
{
    for x in data {
        *x = dist.sample(rng);
    }
}

fn checksum(data: &[f64]) -> f64 {
    data.iter().copied().sum::<f64>().pipe(black_box)
}

fn average(times: &[Duration]) -> Duration {
    let total = times.iter().map(Duration::as_secs_f64).sum::<f64>();
    Duration::from_secs_f64(total / times.len() as f64)
}

fn report(rng_kind: RngKind, tensor: &Timing, sequential: &Timing) {
    println!(
        "{:<12} {:>14} {:>14} {:>14} {:>14} {:>12.3} {:>12.3}",
        rng_kind.name(),
        millis(tensor.best),
        millis(tensor.average),
        millis(sequential.best),
        millis(sequential.average),
        sequential.best.as_secs_f64() / tensor.best.as_secs_f64(),
        sequential.average.as_secs_f64() / tensor.average.as_secs_f64()
    );
    println!(
        "{:<12} tensor checksum {:>16.6}, sequential checksum {:>16.6}",
        "", tensor.checksum, sequential.checksum
    );
}

fn millis(duration: Duration) -> String {
    format!("{:.3} ms", duration.as_secs_f64() * 1_000.0)
}

fn parse_arg(position: usize) -> Option<usize> {
    std::env::args().nth(position)?.parse().ok()
}

fn parse_rng_kinds(position: usize) -> Option<Vec<RngKind>> {
    let arg = std::env::args().nth(position)?;
    if arg.eq_ignore_ascii_case("all") {
        return Some(vec![
            RngKind::Pcg64,
            RngKind::Pcg64Mcg,
            RngKind::SmallRng,
            RngKind::ChaCha8,
            RngKind::ChaCha12,
            RngKind::ChaCha20,
        ]);
    }

    let kinds = arg
        .split(',')
        .filter_map(|name| RngKind::from_name(name.trim()))
        .collect::<Vec<_>>();

    if kinds.is_empty() { None } else { Some(kinds) }
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

trait Pipe: Sized {
    fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}

impl<T> Pipe for T {}

#!/usr/bin/env bash
# Dedicated numerical reference suite plus the direct SIMD correctness tests.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
case "${1:-}" in
    "") ;;
    -h|--help)
        echo 'Usage: bash tests/run_correctness.sh'
        echo 'Runs release reference tests in one- and four-worker pools, then SIMD correctness.'
        echo 'CORRECTNESS_LOG_DIR overrides target/correctness-results.'
        exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo 'Expected no arguments; see --help.' >&2
    exit 2
fi
log_dir="${CORRECTNESS_LOG_DIR:-target/correctness-results}"
mkdir -p -- "$log_dir"
log_file="$log_dir/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
{
    echo "Correctness comparisons started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Log: $log_file"
    rustc --version --verbose
    echo "RUSTFLAGS=${RUSTFLAGS:-}"
    echo "CARGO_ENCODED_RUSTFLAGS=${CARGO_ENCODED_RUSTFLAGS:-}"
    echo 'Criterion: abs(actual - reference) <= atol + rtol * abs(reference)'
    for workers in 1 4; do
        echo "REFERENCE RUN: Rayon workers=$workers; test threads=1"
        RAYON_NUM_THREADS="$workers" cargo test --release --test correctness -- --nocapture --test-threads=1
    done
    echo 'DIRECT SIMD: scalar, AVX2, AVX-512, and automatic dispatch correctness'
    RAYON_NUM_THREADS=4 cargo test --release --lib math::kernels::simd:: -- --nocapture --test-threads=1
    echo "Correctness comparisons completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$log_file"

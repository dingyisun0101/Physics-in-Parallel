#!/usr/bin/env bash
# Run from any directory; retain Cargo's exit status through tee.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

timings=()
case "${1:-}" in
    "") ;;
    --timings) timings=(--include-ignored) ;;
    -h|--help)
        echo 'Usage: bash tests/run_simd.sh [--timings]'
        echo 'Runs release correctness comparisons; --timings adds per-backend measurements.'
        echo 'SIMD_LOG_DIR overrides the default target/simd-results log directory.'
        exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo 'Expected at most one argument; see --help.' >&2
    exit 2
fi

# Explicit intrinsics remain enabled through #[target_feature]. Disable LLVM's
# loop and SLP vectorizers so ordinary Rust reference loops really stay scalar.
# Cargo gives encoded flags precedence, so preserve that format when supplied.
if [[ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]]; then
    export CARGO_ENCODED_RUSTFLAGS+=$'\x1f-Cno-vectorize-loops\x1f-Cno-vectorize-slp'
else
    export RUSTFLAGS="${RUSTFLAGS:-} -Cno-vectorize-loops -Cno-vectorize-slp"
fi
log_dir="${SIMD_LOG_DIR:-target/simd-results}"
mkdir -p -- "$log_dir"
log_file="$log_dir/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
{
    echo "SIMD comparison started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Log: $log_file"
    rustc --version --verbose
    echo "RUSTFLAGS=${RUSTFLAGS:-}"
    echo "CARGO_ENCODED_RUSTFLAGS=${CARGO_ENCODED_RUSTFLAGS:-}"
    echo 'Scalar auto-vectorization: disabled; explicit AVX2/AVX-512: runtime feature guarded'
    # Single test thread keeps summaries readable and timings uncontended.
    cargo test --release --lib math::kernels::simd:: -- --nocapture --test-threads=1 "${timings[@]}"
    echo "SIMD comparison completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$log_file"

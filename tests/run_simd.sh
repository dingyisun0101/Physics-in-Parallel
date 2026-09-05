#!/usr/bin/env bash
# Run from any directory; retain Cargo's exit status through tee.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

timings=()
scalar_baseline=false
for arg in "$@"; do
    case "$arg" in
        --timings) timings=(--include-ignored) ;;
        --no-autovectorize) scalar_baseline=true ;;
        -h|--help)
            echo 'Usage: bash tests/run_simd.sh [--timings] [--no-autovectorize]'
            echo 'Runs release correctness comparisons; --timings adds per-backend measurements.'
            echo 'Normal release auto-vectorization is retained unless --no-autovectorize is supplied.'
            echo 'SIMD_LOG_DIR overrides the default target/simd-results log directory.'
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# Only the opt-in scalar baseline disables LLVM's loop and SLP vectorizers.
# Explicit intrinsics remain enabled through #[target_feature].
# Cargo gives encoded flags precedence, so preserve that format when supplied.
if "$scalar_baseline"; then
    if [[ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]]; then
        export CARGO_ENCODED_RUSTFLAGS+=$'\x1f-Cno-vectorize-loops\x1f-Cno-vectorize-slp'
    else
        export RUSTFLAGS="${RUSTFLAGS:-} -Cno-vectorize-loops -Cno-vectorize-slp"
    fi
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
    if "$scalar_baseline"; then
        echo 'Mode: scalar baseline; LLVM loop and SLP auto-vectorization disabled'
    else
        echo 'Mode: normal release; auto-vectorization allowed unless caller flags override it'
        echo 'Scalar labels identify reference formulas; the compiler may vectorize them'
    fi
    echo 'Explicit AVX2/AVX-512: runtime feature guarded'
    # Single test thread keeps summaries readable and timings uncontended.
    cargo test --release --lib math::kernels::simd:: -- --nocapture --test-threads=1 "${timings[@]}"
    echo "SIMD comparison completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$log_file"

# PiP documentation

These guides describe **Physics in Parallel 4.1.0-alpha**, a breaking alpha release.
Start with the migration notes if you use an earlier version.

| Guide | Contents |
| --- | --- |
| [Getting started](GUIDE.md) | Installation, API layers, containers, randomness, serialization, and particle models |
| [Release and migration notes](RELEASES.md) | Breaking changes, validation fixes, and the release checklist |
| [Examples](EXAMPLES.md) | Complete particle loop and Rust-to-NumPy fixtures |
| [Performance](PERFORMANCE.md) | Release compiler settings, SIMD dispatch, numerical contracts, allocation reuse, and sparse costs |
| [Benchmarks](BENCHMARKS.md) | Reproduction commands, measurements, and limits of the performance evidence |
| [Correctness tests](CORRECTNESS.md) | Independent references, API coverage, tolerances, and terminal logs |
| [SIMD tests](SIMD.md) | AVX2, AVX-512, automatic dispatch, and scalar comparisons |
| [Advanced API](ADVANCED.md) | Raw storage, structured matrices, and generic engines |
| [NumPy helper](NUMPY.md) | Reading validated schema-v2 documents for external analysis |

See the [API reference](https://docs.rs/physics_in_parallel/4.1.0-alpha) for individual
methods and their error, complexity, and output-backend contracts. All shell
commands in these guides run from the repository root unless stated otherwise.

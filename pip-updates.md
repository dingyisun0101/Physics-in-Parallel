# PiP Downstream Update Status

This checklist began as the compatibility review for OmniFluid and
scientific-workflow. The clean-slate decisions are now implemented on PiP's
`tmp` branch. [api.md](api.md) is the normative public contract.

## Implemented

- Universal backend-agnostic `Tensor<T>`, `Matrix<T>`, and `VectorList<T>` with
  explicit dense/sparse construction and representation-preserving Serde.
- Coordinate-based basic access and advanced raw flat-index/storage access.
- Named fallible arithmetic without operator overlap.
- Fully specified `ResolvedRng` across public stochastic APIs.
- Active-Rayon-pool execution with one process-wide optional method cap.
- Direct semantic `PairGenerator` construction without a config DTO.
- Independently meaningful `SquareLatticeGeometry`.
- Canonical `PhysObj` state backed by universal vector lists.
- Model-owned `ParticleStateError` and model-specific source chains.
- Immutable validated laws and transactional pair-keyed networks.
- Private network records and generic interaction/neighbor engines.
- Direct validated Serde without file, JSON-string, payload, or reducer APIs.
- Sealed domain child modules and independent basic/model/advanced preludes.
- Public API contract tests and one current downstream-style example.

## Publication Boundary

PiP 4.0.0 must be validated and published before OmniFluid replaces its local
`../PiP` path dependency with a crates.io version. PiP remains independent of
scientific-workflow; OmniFluid owns workflow seed adaptation and grouped user
configuration.

# PiP Workflow Compatibility Updates

## Purpose

OmniFluid will be refactored as a `scientific-workflow` execution unit that
uses PiP model types directly. This document lists PiP updates needed so
downstream workflow crates can use the `models` API without wrappers,
conversion glue, or ad hoc snapshots.

Assumptions:

- PiP and `scientific-workflow` will be co-developed until their public APIs fit
  together cleanly.
- PiP should remain independent of Workflow policy. It should not own studies,
  phases, persistence cadence, output directories, or terminal UI.
- PiP model types should be usable as Workflow constants, live execution-unit
  internals, and recorded state payloads where that is semantically correct.
- Workflow payloads require `Serialize + Clone + Send + 'static`.
- Workflow execution-unit errors flow through `Box<dyn Error + Send + Sync>`,
  so PiP public errors should implement `std::error::Error`.

## Current Good Fit

The `models` surface already exposes the core pieces OmniFluid wants:

- `PhysObj`
- `SpringNetwork`
- `PowerLawNetwork`
- `ParticleNeighborList`
- `ExplicitEuler`
- `SemiImplicitEuler`
- `LangevinThermostat`
- `ParticleBoundary`
- `KineticEnergyObserver`
- `TemperatureObserver`
- `Spring`
- `PowerLawDecay`
- `RngConfig`
- `IndexedRng`

`PhysObj` is already `Clone + Serialize + Deserialize`, which makes it a good
candidate for direct Workflow state payloads.

`RngConfig` and `IndexedRng` are already `Clone + Copy + Serialize +
Deserialize`, and the indexed design is good for deterministic Workflow seed
derivation because random values do not depend on Rayon scheduling order.

PiP also already avoids owning global workflow policy. Internal parallelism uses
the current Rayon pool, which fits Workflow's role as the higher-level runtime
and resource coordinator.

## Priority 1: Complete Model Error Traits

Problem: many public model errors are `Debug + Clone + PartialEq` but do not
implement `Display` or `std::error::Error`. Downstream Workflow execution units
should be able to use `?` on PiP operations inside `UnitResult`.

Known public errors needing `Display` and `Error`:

- `MassiveParticlesError`
- `IntegratorError`
- `ThermostatError`
- `ObserveError`
- `ParticleBoundaryError`
- `ParticleNeighborListError`
- `SpringNetworkError`
- `PowerLawNetworkError`

Related lower-layer errors used by model errors should also implement
`Display` and `Error`, otherwise source chains cannot be exposed cleanly:

- `AttrsError`
- `InteractionError`
- `NeighborListError`
- `BoundaryError`
- `VectorSamplingError`
- `TensorRandError`
- `SquareLatticeConfigError`
- `PairGenerationError`
- `KernelError`
- `ScalarCastError` already implements `Error` but should be checked for
  consistent `Display`

Recommended approach:

1. Add hand-written `Display` impls for all public errors.
2. Add `impl std::error::Error` for all public errors.
3. Where an error wraps another public error, implement `source()`.
4. Add tests that coerce each public model error into
   `Box<dyn std::error::Error + Send + Sync>`.

Exit criteria:

- downstream code can write `pip_call()?` inside a Workflow `UnitResult`
- errors preserve useful context and source chains
- no Workflow execution unit needs to stringify PiP errors manually

## Priority 2: Serialize Model Laws And Networks

Problem: Workflow can record `PhysObj` directly, but model laws and interaction
networks are not yet serializable as public payloads. OmniFluid can work around
that with its own `Vec<SpringPair>` snapshot, but the clean goal is to record
PiP-owned model state directly where PiP owns the model type.

Types that should derive or implement Serde:

- `Spring`
- `PowerLawDecay`
- `InteractionOrder`
- `InteractionNodes`
- `InteractionTopology`
- `Interaction<T>` where `T: Serialize + DeserializeOwned`
- `SpringNetwork`
- `PowerLawNetwork`
- `NeighborList`
- `ParticleNeighborList`
- `ParticleSelection`
- integrator selector/config types introduced in a later step
- thermostat selector/config types introduced in a later step

Recommended serialization policy:

- Use versioned payloads for persistent model state where format stability
  matters.
- Keep topology and payload storage synchronized during deserialization.
- Validate spring and power-law payloads while deserializing networks.
- Reject malformed node lists, invalid object ids, invalid arity, duplicate
  unordered pairs, and invalid payload values.
- Preserve deterministic iteration order in serialized output where possible.

Exit criteria:

- `SpringNetwork` can be inserted into Workflow `SystemState` directly when a
  simulation wants to record topology and spring parameters.
- `PowerLawNetwork` has the same capability.
- Tests round-trip non-empty networks and reject malformed serialized networks.

## Priority 3: Add Stable Snapshot And Iteration APIs

Problem: even with Serde, downstream crates often need lightweight diagnostic
views without depending on engine internals. Current accessors expose
`Interaction<T>`, but Workflow integrations should have stable model-level
export APIs.

Recommended additions:

- `SpringRecord { i, j, spring }`
- `PowerLawRecord { i, j, law }`
- `SpringNetwork::records() -> Vec<SpringRecord>`
- `SpringNetwork::from_records(num_particles, records) -> Result<Self, ...>`
- `PowerLawNetwork::records() -> Vec<PowerLawRecord>`
- `PowerLawNetwork::from_records(num_particles, records) -> Result<Self, ...>`
- optional borrowed iterators over `(i, j, &Spring)` and
  `(i, j, &PowerLawDecay)`

These records should be `Clone + Debug + PartialEq + Serialize + Deserialize`.

Exit criteria:

- downstream crates can record compact interaction views without touching
  `Interaction<T>` internals
- network reconstruction is validated and deterministic
- OmniFluid can choose between direct network payloads and compact records

## Priority 4: Owned Config Types For Workflow Constants

Problem: some PiP user-facing configuration enums borrow slices, which is good
for low-allocation execution calls but not suitable as Workflow constants,
because Workflow constants must be owned and `DeserializeOwned + 'static`.

Borrowed examples:

- `VectorSamplingMethod<'a>`
- `VelocitySamplingMethod<'a>`

Recommended additions:

- `VectorSamplingConfig`
- `VelocitySamplingConfig`
- conversion methods such as `as_method()` or direct `sample_*_config` helpers
- serde support with `deny_unknown_fields`

Example direction:

```rust
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum VelocitySamplingConfig {
    Uniform { low: f64, high: f64 },
    MaxwellBoltzmann { tau: f64 },
    GaussianPerAxis { mean: Vec<f64>, std: Vec<f64> },
}
```

Exit criteria:

- Workflow `parameters.json` can express PiP sampling choices directly
- downstream crates do not need to invent duplicate owned config enums
- borrowed low-level methods can remain for allocation-sensitive internal calls

## Priority 5: Configurable Integrators And Thermostats

Problem: PiP exposes concrete integrator and thermostat types, but downstream
Workflow projects need serializable configuration selectors for choosing them
from JSON.

Recommended additions:

- `IntegratorConfig`
- `IntegratorKind`
- `IntegratorConfig::build() -> Box<dyn Integrator + Send>`
- `ThermostatConfig`
- `ThermostatKind`
- `ThermostatConfig::build(selection) -> Result<Box<dyn Thermostat + Send>, ...>`

Initial integrator kinds:

- `explicit_euler`
- `semi_implicit_euler`

Future PiP integrators can be added as new variants. Because PiP is pre-1.0,
breaking enum changes are acceptable when scientifically justified, but versioned
or non-exhaustive config patterns should be considered.

Exit criteria:

- Workflow constants can select PiP integrators without OmniFluid-specific
  wrappers
- future PiP integrators can be consumed by OmniFluid through configuration
- thermostat configuration records resolved RNG provenance

## Priority 6: Make Langevin Thermostat Fully Recordable

Problem: `LangevinThermostat` is deterministic and exposes its resolved
`RngConfig` and `step_counter`, but it is not currently serializable. Workflow
recordings and restart-like downstream workflows may need to record the
thermostat state alongside particle state.

Recommended additions:

- derive or implement `Serialize + Deserialize` for `LangevinThermostat`
- ensure `ParticleSelection` is serializable first
- validate `tau_target`, `gamma`, RNG config, and step counter during
  deserialization
- consider a public constructor for restoring from resolved state

Exit criteria:

- a stochastic thermostat can be recorded and reconstructed without losing RNG
  position/provenance
- serialized state is sufficient to continue the same stochastic sequence

## Priority 7: Public Particle Initialization Helpers

Problem: `create_template`, `randomize_r`, and `randomize_v` are useful pieces,
but downstream model crates still commonly need a higher-level, serializable
particle-initialization config that composes template creation, position
sampling, velocity sampling, mass assignment, alive/rigid defaults, and RNG
provenance.

Recommended additions:

- `MassiveParticlesConfig`
- `MassDistributionConfig`
- `create_particles(config, rngs) -> Result<PhysObj, MassiveParticlesError>`
- optional helpers for setting all masses and inverse masses consistently
- optional helper for adding custom attributes with model-level validation

Exit criteria:

- OmniFluid can build PiP particle state from Workflow constants without its own
  particle wrapper
- mass assignment is a PiP model concern rather than a downstream reimplementation
- all resolved randomness can be returned for provenance

## Priority 8: Boundary Config Compatibility

Problem: particle boundary behavior composes with `ContinuousBoundary`, but
Workflow constants need owned, serializable boundary config. The concrete
continuous boundary structs also do not currently derive Serde.

Recommended additions:

- serde support for `PeriodicBox`, `ClampBox`, and `ReflectBox`
- `BoundaryConfig` or `ContinuousBoundaryConfig`
- builder methods returning boxed or enum-owned boundary values
- `ParticleBoundaryConfig` if particle-specific selection/mask behavior grows

Exit criteria:

- Workflow constants can configure PiP boundary behavior directly
- downstream crates do not need their own duplicate boundary enums

## Priority 9: Threading Contract Tests

Problem: Workflow owns the runtime compute budget. PiP should continue using the
ambient Rayon pool and should not create hidden global pools in model code.

Recommended additions:

- tests or documentation confirming model operations use ambient Rayon
- avoid environment-variable-driven model behavior that bypasses Workflow
  scheduling
- keep `ComputePool` as explicit opt-in convenience, not a hidden model default

Exit criteria:

- Workflow can coordinate PiP model execution without oversubscription surprises

## Priority 10: Workflow-Facing Examples

Problem: once the above compatibility items exist, PiP should prove them through
small examples that mirror downstream Workflow use without depending on
OmniFluid.

Recommended additions:

- example that creates a `PhysObj`, `SpringNetwork`, integrator, thermostat, and
  observer from serializable configs
- example that serializes and deserializes full particle and network state
- example that uses an externally supplied seed as the only stochastic input
- optional dev-only example showing insertion of PiP payloads into a
  `scientific-workflow::SystemState`

Exit criteria:

- downstream crates have a clear reference for PiP-native Workflow integration
- PiP remains policy-free while demonstrating compatibility

## Suggested Implementation Order

1. Complete `Display` and `Error` impls for all public model and model-adjacent
   errors.
2. Add Serde to law payloads and simple enums.
3. Add serializable record types for spring and power-law networks.
4. Add validated Serde for `Interaction<T>`, `SpringNetwork`, and
   `PowerLawNetwork`.
5. Add owned sampling config types.
6. Add integrator and thermostat config selectors.
7. Add serializable Langevin thermostat state.
8. Add high-level particle initialization config and helpers.
9. Add serializable boundary config.
10. Add Workflow-facing examples/tests.

## Expected OmniFluid Impact

After these PiP updates, OmniFluid can:

- store live particles as PiP `PhysObj`
- store live springs as PiP `SpringNetwork`
- select PiP integrators from Workflow constants
- select PiP thermostats from Workflow constants
- use Workflow-derived seeds through PiP `RngConfig`
- record PiP model state directly when appropriate
- avoid particle and spring wrapper types whose only job is filling gaps in PiP

OmniFluid will still own mask semantics, noise semantics, topology policies,
Workflow execution-unit composition, and scientific analysis meaning.

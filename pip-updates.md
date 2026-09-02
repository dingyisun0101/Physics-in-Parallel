# PiP Downstream API Updates

## Purpose

OmniFluid uses PiP as a general physics library. This checklist records changes
that improve PiP's independent public API while allowing configuration-driven
downstream crates to construct, execute, inspect, and record PiP models.

PiP does not know about `scientific-workflow`. OmniFluid owns its complete
experiment schema and translates validated values into ordinary PiP
constructors.

## Ownership Boundary

PiP owns:

- physical laws and validated law values
- particle state and canonical particle operations
- interaction topology and model networks
- integrators, thermostats, boundaries, and observers
- deterministic random-number primitives used by its operations
- errors, constructors, accessors, and snapshots for those concepts

Downstream crates own:

- configuration DTOs and file formats
- Serde-tagged selectors for choosing PiP implementations
- defaults, aliases, and cross-component policy
- construction order and translation into PiP calls
- workflow orchestration, persistence cadence, and provenance policy

Adding a PiP config type is not part of this work. Serde belongs on a PiP type
only when the type itself is meaningful physical state or a stable physical
value, not merely because a downstream application wants to deserialize input.

## Current Public Surface

The model API already exposes the principal pieces OmniFluid needs:

- `PhysObj`
- `SpringNetwork` and `PowerLawNetwork`
- `ParticleNeighborList`
- `ExplicitEuler` and `SemiImplicitEuler`
- `LangevinThermostat`
- `ParticleBoundary`
- kinetic-energy and temperature observers
- validated `Spring` and `PowerLawDecay` laws
- `RngConfig` and `IndexedRng`

`PhysObj`, `RngConfig`, and `IndexedRng` already support Serde. PiP also uses
the ambient Rayon pool in model operations, so a downstream runtime can own the
thread budget without PiP creating hidden pools.

## Stage 2: Error Ergonomics

Status: completed on `tmp`.

Goal: every public model operation returns an error that composes naturally in
ordinary Rust applications.

Public model errors needing `Display` and `std::error::Error`:

- `MassiveParticlesError`
- `IntegratorError`
- `ThermostatError`
- `ObserveError`
- `ParticleBoundaryError`
- `ParticleNeighborListError`
- `SpringNetworkError`
- `PowerLawNetworkError`

Public lower-layer errors reached through those models need the same treatment:

- `AttrsError`
- `InteractionError`
- `NeighborListError`
- `BoundaryError`
- `VectorSamplingError`

Implementation rules:

1. Write concise, contextual `Display` messages containing relevant labels,
   dimensions, indices, and invalid values.
2. Return wrapped errors from `source()` instead of flattening them into debug
   strings.
3. Preserve the existing typed variants unless source preservation requires a
   clearer variant.
4. Verify all exported errors are `Send + Sync + std::error::Error + 'static`.
5. Test representative display text and complete source chains.

No error should mention Workflow or a downstream configuration format.

## Stage 3: Serializable Model State

Status: completed on `tmp`.

Goal: serialize physical values and model state through validated, deterministic
representations.

Planned public values:

- Serde-enabled `Spring`, `SpringCutoff`, `PowerLawDecay`, and `PowerLawRange`
- Serde-enabled `InteractionOrder`, `InteractionNodes`, and validated
  `InteractionTopology`
- `SpringRecord { i, j, spring }`
- `PowerLawRecord { i, j, law }`
- deterministic record iterators or exports on both network types
- validated `from_records` constructors
- custom Serde for `SpringNetwork` and `PowerLawNetwork` through their records

Model-network serialization rules:

1. Serialize only active interactions, never hash-map layout, spare capacity,
   free-list order, or another implementation detail.
2. Emit records in stable interaction-id order.
3. Include the particle bound independently of the largest active endpoint so
   empty networks and unused particles round-trip correctly.
4. Reconstruct through public validation paths.
5. Reject self-pairs, out-of-bound endpoints, duplicate unordered pairs, and
   invalid law values with typed network errors.
6. Keep configuration selectors and construction policy downstream.

Generic `InteractionTopology` persistence preserves active interaction ids and
explicit holes. Deserialization canonicalizes nodes and rebuilds the lookup map
and reusable-id list; hash-map layout and free-list ordering are never part of
the serialized representation.

## Stage 4: Construction And Inspection APIs

Goal: downstream crates can translate their own validated configuration into
PiP models and inspect those models without private implementation knowledge.

Audit targets:

- canonical particle template construction
- spring and power-law network bounds, records, and iteration
- continuous and particle boundary constructors and geometry accessors
- direct integrator construction and execution
- thermostat construction, state inspection, and deterministic restoration
- observer construction and selection inspection

Implementation rules:

1. Constructors accept direct semantic parameters or existing PiP domain
   values, not configuration DTOs.
2. Accessors expose stable physical meaning, not backing-container layout.
3. Restore constructors validate the same invariants as fresh constructors.
4. Borrowed sampling APIs remain borrowed; downstream owned configs can lend
   slices when invoking them.
5. Add API-surface tests that construct and inspect models as an external crate
   would through `prelude::models`.

Stage 4 deliberately does not add integrator selectors, thermostat selectors,
boundary configs, particle configs, or Workflow examples.

## Deferred Work

The following work starts only after the Stage 4 public API and error review:

- higher-level particle construction helpers that preserve physical invariants
- additional PiP integrators
- thermostat checkpoint semantics beyond the current model
- boundary and neighbor-list improvements discovered by OmniFluid integration
- ambient-threading contract tests and documentation
- generic host-integration examples

Each deferred change must pass the same test: it should improve PiP as an
independent physics crate even if OmniFluid and `scientific-workflow` did not
exist.

## Expected OmniFluid Use

After Stages 2 through 4, OmniFluid can:

- deserialize its own grouped experiment configuration
- validate and resolve that configuration through OmniFluid accessors
- call PiP constructors with direct physical parameters
- store live particles and spring networks as PiP types
- choose integrators and thermostats using OmniFluid-owned enums
- pass borrowed slices from owned OmniFluid sampling configs
- record PiP physical state directly where its state representation is stable
- propagate PiP errors without string conversion

OmniFluid continues to own mask semantics, noise semantics, topology policy,
workflow integration, and scientific interpretation.

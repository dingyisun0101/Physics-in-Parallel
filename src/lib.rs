/*!
Physics in Parallel provides backend-agnostic numerical containers, spatial
tools, and reusable physical models.

# Alpha warning

PiP 4.0 is an unstable, clean-slate rewrite. Alpha releases are incompatible
with PiP 3.x, and their public API and serialized representations may change
without migration support before 4.0.0.

# Thread-pool responsibility

PiP uses the active Rayon pool and never creates one. Applications must
configure one shared pool before concurrent work begins or let their workflow
runner do so. Independent pools can oversubscribe the machine. Normally call
[`threading::set_max_threads`] once during startup to set the maximum worker
participation requested by any single PiP method.

# API tiers

Domain roots are authoritative. Independent convenience preludes expose basic
numerics, ready physical models, and opt-in advanced facilities:

```rust
use physics_in_parallel::prelude::basic::*;
use physics_in_parallel::prelude::models::*;
use physics_in_parallel::prelude::advanced::*;
```

Start with the basic or model API. Raw storage and generic engines are advanced;
implementation modules are private. PiP owns runtime scientific values, while
applications own user configuration, persistence policy, and workflow
integration.
*/

pub mod advanced;
pub(crate) mod engines;
pub mod math;
pub mod models;
pub mod prelude;
pub mod rng;
pub(crate) mod sampling;
pub mod space;
#[path = "parallel.rs"]
pub mod threading;

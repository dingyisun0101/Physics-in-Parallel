/*!
Physics in Parallel crate root.

Purpose:
This crate provides layered infrastructure for physics-oriented numerical
simulation:

- `math` defines scalar, tensor, matrix, vector-list, random-fill, and math IO
  foundations.
- `space` adds continuous and discrete spatial semantics on top of math
  containers.
- `engines` provides model-agnostic runtime storage and interaction backends.
- `models` provides concrete physical model pieces that use the lower layers.

The crate-wide prelude separates foundational, model-level, and advanced APIs:

`use physics_in_parallel::prelude::basic::*;`
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

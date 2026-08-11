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

The crate-wide prelude re-exports the common user-facing API from all ready
layers:

`use physics_in_parallel::prelude::*;`
*/

pub mod engines;
pub mod io;
pub mod math;
pub mod models;
pub mod prelude;
pub mod rng;
pub mod space;

/*!
Square-lattice discrete spaces.

Purpose:
    This module groups the data representation and lattice-specific random
    displacement tools for square, cubic, and hypercubic lattices. Keeping
    these files under `square_lattice` leaves room for other discrete-space
    families, such as triangular or graph-based spaces, to define their own
    representation and displacement semantics later.
*/

pub mod displacement;
pub mod representation;

pub use displacement::RandPairGenerator;
pub use representation::{
    BoundaryCondition, SquareLattice, SquareLatticeConfig, SquareLatticeInitMethod, VacancyValue,
};

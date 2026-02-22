/*!
Structure-of-arrays engine implementation.
*/

pub mod phys_obj;
pub mod interaction;
pub mod integrator;
pub mod boundary;
pub mod neighbors;
pub mod observe;
pub mod thermostat;

// Canonical SoA exports.
pub use boundary::{Boundary, BoundaryError, ClampBox, PeriodicBox, ReflectBox};
pub use integrator::{ExplicitEuler, Integrator, IntegratorError, SemiImplicitEuler};
pub use interaction::{EdgeId, Interaction, InteractionError, PairTopology};
pub use neighbors::{AllPairs, NeighborError, NeighborProvider, RadiusPairs};
pub use observe::{
    KineticEnergyObserver,
    MeanReducer,
    ObserveError,
    Observer,
    Reducer,
    TemperatureObserver,
};
pub use phys_obj::{AttrId, AttrMeta, AttrSchema, ObjId, ObjKind, PhysObj, PhysObjError};
pub use thermostat::langevin::LangevinThermostat;
pub use thermostat::{Thermostat, ThermostatError};

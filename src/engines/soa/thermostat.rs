/*!
Thermostat interfaces and implementations for `engines::soa`.
*/

pub mod langevin;

use super::phys_obj::{PhysObj, PhysObjError};

/// Errors returned by thermostat modules.
#[derive(Debug, Clone, PartialEq)]
pub enum ThermostatError {
    /// Parameter must be finite and within expected range.
    InvalidParam { field: &'static str, value: f64 },
    /// Time-step must be finite and strictly positive.
    InvalidDt { dt: f64 },
    /// Lifted error from `PhysObj` accessors.
    PhysObj(PhysObjError),
}

impl From<PhysObjError> for ThermostatError {
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `from` logic for this module.
    /// - Parameters:
    ///   - `value` (`PhysObjError`): Value provided by caller for write/update behavior.
    fn from(value: PhysObjError) -> Self {
        Self::PhysObj(value)
    }
}

/// Generic thermostat contract.
pub trait Thermostat<const D: usize> {
    /// Annotation:
    /// - Purpose: Executes `apply` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj<D>`): Object-state container used by this operation.
    ///   - `dt` (`f64`): Time-step used for one simulation/integration update.
    fn apply(&mut self, objects: &mut PhysObj<D>, dt: f64) -> Result<(), ThermostatError>;
}

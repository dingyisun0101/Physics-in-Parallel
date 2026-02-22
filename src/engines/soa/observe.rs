/*!
Generic observer/reducer interfaces for `engines::soa`.
*/

use super::phys_obj::{PhysObj, PhysObjError};

/// Errors returned by observers.
#[derive(Debug, Clone, PartialEq)]
pub enum ObserveError {
    /// Lifted error from `PhysObj` accessors.
    PhysObj(PhysObjError),
    /// Invalid state encountered during measurement.
    InvalidState { field: &'static str, value: f64 },
}

impl From<PhysObjError> for ObserveError {
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `from` logic for this module.
    /// - Parameters:
    ///   - `value` (`PhysObjError`): Value provided by caller for write/update behavior.
    fn from(value: PhysObjError) -> Self {
        Self::PhysObj(value)
    }
}

/// Generic observable contract.
pub trait Observer<const D: usize> {
    type Output;
    /// Annotation:
    /// - Purpose: Executes `observe` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&PhysObj<D>`): Object-state container used by this operation.
    fn observe(&self, objects: &PhysObj<D>) -> Result<Self::Output, ObserveError>;
}

/// Generic reduction contract over measured values.
pub trait Reducer<T> {
    /// Annotation:
    /// - Purpose: Executes `reduce` logic for this module.
    /// - Parameters:
    ///   - `values` (`&[T]`): Parameter of type `&[T]` used by `reduce`.
    fn reduce(&self, values: &[T]) -> T;
}

/// Mean reducer for `f64` sequences.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanReducer;

impl Reducer<f64> for MeanReducer {
    /// Annotation:
    /// - Purpose: Executes `reduce` logic for this module.
    /// - Parameters:
    ///   - `values` (`&[f64]`): Parameter of type `&[f64]` used by `reduce`.
    fn reduce(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / (values.len() as f64)
    }
}

/// Total kinetic energy observer.
#[derive(Debug, Clone, Copy)]
pub struct KineticEnergyObserver {
    pub include_dead: bool,
}

impl Default for KineticEnergyObserver {
    /// Annotation:
    /// - Purpose: Executes `default` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    ///   - (none): This function takes no explicit parameters.
    fn default() -> Self {
        Self { include_dead: false }
    }
}

impl<const D: usize> Observer<D> for KineticEnergyObserver {
    type Output = f64;

    /// Annotation:
    /// - Purpose: Executes `observe` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&PhysObj<D>`): Object-state container used by this operation.
    fn observe(&self, objects: &PhysObj<D>) -> Result<Self::Output, ObserveError> {
        let n = objects.len();
        let mut ke = 0.0;

        for i in 0..n {
            if !self.include_dead && !objects.is_alive(i)? {
                continue;
            }

            let inv_mass = objects.inv_mass_of(i)?;
            if !inv_mass.is_finite() || inv_mass <= 0.0 {
                return Err(ObserveError::InvalidState {
                    field: "inv_mass",
                    value: inv_mass,
                });
            }

            let v = objects.vel_of(i)?;
            let mut v2 = 0.0;
            for &vd in v {
                v2 += vd * vd;
            }

            // m = 1 / inv_mass
            ke += 0.5 * v2 / inv_mass;
        }

        Ok(ke)
    }
}

/// Kinetic temperature-like scalar: `2 * KE / (N_alive * D)`.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureObserver {
    pub include_dead: bool,
}

impl Default for TemperatureObserver {
    /// Annotation:
    /// - Purpose: Executes `default` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    ///   - (none): This function takes no explicit parameters.
    fn default() -> Self {
        Self { include_dead: false }
    }
}

impl<const D: usize> Observer<D> for TemperatureObserver {
    type Output = f64;

    /// Annotation:
    /// - Purpose: Executes `observe` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&PhysObj<D>`): Object-state container used by this operation.
    fn observe(&self, objects: &PhysObj<D>) -> Result<Self::Output, ObserveError> {
        let ke = KineticEnergyObserver {
            include_dead: self.include_dead,
        }
        .observe(objects)?;

        let n = objects.len();
        let mut count = 0usize;
        for i in 0..n {
            if self.include_dead || objects.is_alive(i)? {
                count += 1;
            }
        }

        if count == 0 {
            return Ok(0.0);
        }

        Ok((2.0 * ke) / ((count * D) as f64))
    }
}

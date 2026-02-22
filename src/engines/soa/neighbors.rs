/*!
Neighbor candidate providers for `engines::soa`.
*/

use super::phys_obj::{ObjId, PhysObj, PhysObjError};

/// Errors returned by neighbor providers.
#[derive(Debug, Clone, PartialEq)]
pub enum NeighborError {
    /// Radius cutoff must be finite and strictly positive.
    InvalidCutoff { cutoff: f64 },
    /// Lifted error from `PhysObj` accessors.
    PhysObj(PhysObjError),
}

impl From<PhysObjError> for NeighborError {
    #[inline]
    /// Annotation:
    /// - Purpose: Executes `from` logic for this module.
    /// - Parameters:
    ///   - `value` (`PhysObjError`): Value provided by caller for write/update behavior.
    fn from(value: PhysObjError) -> Self {
        Self::PhysObj(value)
    }
}

/// Produces pair-candidate lists for topology updates or interaction evaluation.
pub trait NeighborProvider<const D: usize> {
    /// Annotation:
    /// - Purpose: Executes `candidates` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&PhysObj<D>`): Object-state container used by this operation.
    fn candidates(&self, objects: &PhysObj<D>) -> Result<Vec<(ObjId, ObjId)>, NeighborError>;
}

/// Full `i<j` scan.
#[derive(Debug, Clone, Copy)]
pub struct AllPairs {
    pub include_dead: bool,
}

impl Default for AllPairs {
    /// Annotation:
    /// - Purpose: Executes `default` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    ///   - (none): This function takes no explicit parameters.
    fn default() -> Self {
        Self { include_dead: false }
    }
}

impl<const D: usize> NeighborProvider<D> for AllPairs {
    /// Annotation:
    /// - Purpose: Executes `candidates` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&PhysObj<D>`): Object-state container used by this operation.
    fn candidates(&self, objects: &PhysObj<D>) -> Result<Vec<(ObjId, ObjId)>, NeighborError> {
        let n = objects.len();
        let mut out = Vec::with_capacity(n.saturating_mul(n.saturating_sub(1)) / 2);

        for i in 0..n {
            if !self.include_dead && !objects.is_alive(i)? {
                continue;
            }
            for j in (i + 1)..n {
                if !self.include_dead && !objects.is_alive(j)? {
                    continue;
                }
                out.push((i, j));
            }
        }
        Ok(out)
    }
}

/// Radius-filtered pair scan.
#[derive(Debug, Clone, Copy)]
pub struct RadiusPairs {
    cutoff: f64,
    include_dead: bool,
}

impl RadiusPairs {
    /// Annotation:
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    ///   - `cutoff` (`f64`): Parameter of type `f64` used by `new`.
    ///   - `include_dead` (`bool`): Parameter of type `bool` used by `new`.
    pub fn new(cutoff: f64, include_dead: bool) -> Result<Self, NeighborError> {
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(NeighborError::InvalidCutoff { cutoff });
        }
        Ok(Self {
            cutoff,
            include_dead,
        })
    }

    #[inline]
    /// Annotation:
    /// - Purpose: Executes `cutoff` logic for this module.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl<const D: usize> NeighborProvider<D> for RadiusPairs {
    /// Annotation:
    /// - Purpose: Executes `candidates` logic for this module.
    /// - Parameters:
    ///   - `objects` (`&PhysObj<D>`): Object-state container used by this operation.
    fn candidates(&self, objects: &PhysObj<D>) -> Result<Vec<(ObjId, ObjId)>, NeighborError> {
        let n = objects.len();
        let cutoff_sq = self.cutoff * self.cutoff;
        let mut out = Vec::new();

        for i in 0..n {
            if !self.include_dead && !objects.is_alive(i)? {
                continue;
            }
            let ri = objects.pos_of(i)?;

            for j in (i + 1)..n {
                if !self.include_dead && !objects.is_alive(j)? {
                    continue;
                }

                let rj = objects.pos_of(j)?;
                let mut d2 = 0.0;
                for d in 0..D {
                    let dr = rj[d] - ri[d];
                    d2 += dr * dr;
                }
                if d2 <= cutoff_sq {
                    out.push((i, j));
                }
            }
        }

        Ok(out)
    }
}

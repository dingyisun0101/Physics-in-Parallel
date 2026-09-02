/*!
Shared interpretation helpers for canonical particle state.

Purpose:
This module centralizes the repeated logic for reading canonical particle
attributes from `PhysObj`. It validates vector/scalar attribute shape, gathers
alive and rigid masks, and reads inverse-mass values. Higher-level modules keep
their own public error types and convert from `ParticleStateError`.
*/

use core::fmt;

use rayon::prelude::*;

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{
    ATTR_ALIVE, ATTR_M_INV, ATTR_RIGID, ParticleSelection, is_alive_value, is_rigid_value,
};

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParticleStateError {
    MissingAttribute {
        label: String,
    },
    WrongScalarType {
        label: String,
        expected: String,
        actual: String,
    },
    InvalidAttrShape {
        label: &'static str,
        expected_dim: usize,
        got_dim: usize,
    },
    InconsistentParticleCount {
        label: &'static str,
        expected: usize,
        got: usize,
    },
    ParticleOutOfBounds {
        label: String,
        particle: usize,
        particle_count: usize,
    },
    InvalidAttributeStorage {
        message: String,
    },
}

impl From<AttrsError> for ParticleStateError {
    fn from(value: AttrsError) -> Self {
        match value {
            AttrsError::UnknownLabel { label } => Self::MissingAttribute { label },
            AttrsError::WrongType {
                label,
                expected,
                got,
            } => Self::WrongScalarType {
                label,
                expected,
                actual: got,
            },
            AttrsError::ObjOutOfBounds { label, obj, n } => Self::ParticleOutOfBounds {
                label,
                particle: obj,
                particle_count: n,
            },
            error => Self::InvalidAttributeStorage {
                message: error.to_string(),
            },
        }
    }
}

impl fmt::Display for ParticleStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingAttribute { label } => {
                write!(formatter, "particle state is missing attribute `{label}`")
            }
            Self::WrongScalarType {
                label,
                expected,
                actual,
            } => write!(
                formatter,
                "particle attribute `{label}` has scalar type `{actual}`; expected `{expected}`"
            ),
            Self::InvalidAttrShape {
                label,
                expected_dim,
                got_dim,
            } => write!(
                formatter,
                "particle attribute `{label}` has dimension {got_dim}; expected {expected_dim}"
            ),
            Self::InconsistentParticleCount {
                label,
                expected,
                got,
            } => write!(
                formatter,
                "particle attribute `{label}` has {got} rows; expected {expected}"
            ),
            Self::ParticleOutOfBounds {
                label,
                particle,
                particle_count,
            } => write!(
                formatter,
                "particle {particle} is outside attribute `{label}` with {particle_count} rows"
            ),
            Self::InvalidAttributeStorage { message } => {
                write!(formatter, "invalid particle attribute storage: {message}")
            }
        }
    }
}

impl std::error::Error for ParticleStateError {}

#[derive(Debug, Clone, Default)]
pub(crate) struct ParticleMasks {
    pub alive: Option<Vec<bool>>,
    pub rigid: Option<Vec<bool>>,
}

impl ParticleMasks {
    #[inline]
    pub fn should_skip(&self, i: usize) -> bool {
        self.alive.as_ref().is_some_and(|flags| !flags[i])
            || self.rigid.as_ref().is_some_and(|flags| flags[i])
    }

    #[inline]
    pub fn is_included(&self, selection: ParticleSelection, i: usize) -> bool {
        selection.includes_dead() || self.alive.as_ref().is_none_or(|flags| flags[i])
    }

    pub fn included_count(&self, selection: ParticleSelection, n: usize) -> usize {
        if selection.includes_dead() {
            return n;
        }
        self.alive
            .as_ref()
            .map_or(n, |flags| flags.par_iter().filter(|&&alive| alive).count())
    }
}

pub(crate) fn validate_vector_attr_f64(
    objects: &PhysObj,
    label: &'static str,
    expected_dim: usize,
    expected_n: usize,
) -> Result<(), ParticleStateError> {
    let attr = objects.core.get::<f64>(label)?;
    validate_attr_shape(label, attr.dim(), expected_dim)?;
    validate_attr_count(label, attr.num_vectors(), expected_n)?;
    Ok(())
}

pub(crate) fn validate_scalar_shape(
    label: &'static str,
    got_dim: usize,
    got_n: usize,
    expected_n: usize,
) -> Result<(), ParticleStateError> {
    validate_attr_shape(label, got_dim, 1)?;
    validate_attr_count(label, got_n, expected_n)?;
    Ok(())
}

pub(crate) fn gather_inverse_mass(
    objects: &PhysObj,
    n: usize,
) -> Result<Vec<f64>, ParticleStateError> {
    let m_inv = objects.core.get::<f64>(ATTR_M_INV)?;
    validate_scalar_shape(ATTR_M_INV, m_inv.dim(), m_inv.num_vectors(), n)?;
    Ok((0..n).map(|i| m_inv.get(i as isize, 0)).collect())
}

pub(crate) fn gather_alive_flags(
    objects: &PhysObj,
    n: usize,
    selection: ParticleSelection,
) -> Result<Option<Vec<bool>>, ParticleStateError> {
    if selection.includes_dead() || !objects.core.contains(ATTR_ALIVE) {
        return Ok(None);
    }

    let alive = objects.core.get::<u8>(ATTR_ALIVE)?;
    validate_scalar_shape(ATTR_ALIVE, alive.dim(), alive.num_vectors(), n)?;
    Ok(Some(
        (0..n)
            .map(|i| is_alive_value(alive.get(i as isize, 0)))
            .collect(),
    ))
}

pub(crate) fn gather_rigid_flags(
    objects: &PhysObj,
    n: usize,
) -> Result<Option<Vec<bool>>, ParticleStateError> {
    if !objects.core.contains(ATTR_RIGID) {
        return Ok(None);
    }

    let rigid = objects.core.get::<u8>(ATTR_RIGID)?;
    validate_scalar_shape(ATTR_RIGID, rigid.dim(), rigid.num_vectors(), n)?;
    Ok(Some(
        (0..n)
            .map(|i| is_rigid_value(rigid.get(i as isize, 0)))
            .collect(),
    ))
}

pub(crate) fn gather_masks(
    objects: &PhysObj,
    n: usize,
    selection: ParticleSelection,
) -> Result<ParticleMasks, ParticleStateError> {
    Ok(ParticleMasks {
        alive: gather_alive_flags(objects, n, selection)?,
        rigid: gather_rigid_flags(objects, n)?,
    })
}

#[inline]
fn validate_attr_shape(
    label: &'static str,
    got_dim: usize,
    expected_dim: usize,
) -> Result<(), ParticleStateError> {
    if got_dim != expected_dim {
        return Err(ParticleStateError::InvalidAttrShape {
            label,
            expected_dim,
            got_dim,
        });
    }
    Ok(())
}

#[inline]
fn validate_attr_count(
    label: &'static str,
    got: usize,
    expected: usize,
) -> Result<(), ParticleStateError> {
    if got != expected {
        return Err(ParticleStateError::InconsistentParticleCount {
            label,
            expected,
            got,
        });
    }
    Ok(())
}

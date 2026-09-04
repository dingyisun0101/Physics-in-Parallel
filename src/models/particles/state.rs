/*!
Shared interpretation helpers for canonical particle state.

Purpose:
This module centralizes the repeated logic for reading canonical particle
attributes from `PhysObj`. It validates vector/scalar attribute shape, gathers
alive and rigid masks, and reads inverse-mass values. Higher-level modules keep
their own public error types and convert from `ParticleStateError`.
*/

pub use crate::engines::soa::phys_obj::ParticleStateError;
use crate::engines::soa::phys_obj::PhysObj;
use crate::models::particles::attrs::{ATTR_ALIVE, ATTR_M_INV, ATTR_RIGID, ParticleSelection};

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
    Ok((0..n)
        .map(|i| m_inv.get(i, 0).expect("validated scalar coordinates"))
        .collect())
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

/// Short-lived mask snapshot borrowing dense storage without boolean gathers.
pub(crate) struct BorrowedMasks<'a> {
    alive: Option<std::borrow::Cow<'a, [u8]>>,
    rigid: Option<std::borrow::Cow<'a, [u8]>>,
}

impl<'a> BorrowedMasks<'a> {
    pub(crate) fn new(
        flags: [Option<&'a crate::math::VectorList<u8>>; 2],
        n: usize,
    ) -> Result<Self, ParticleStateError> {
        for (flag, label) in flags.iter().zip([ATTR_ALIVE, ATTR_RIGID]) {
            if let Some(flag) = flag {
                validate_scalar_shape(label, flag.dim(), flag.num_vectors(), n)?;
            }
        }
        Ok(Self {
            alive: flags[0].map(|flag| flag.borrow_values()),
            rigid: flags[1].map(|flag| flag.borrow_values()),
        })
    }
    pub(crate) fn alive(&self, index: usize) -> bool {
        self.alive.as_ref().is_none_or(|flags| flags[index] != 0)
    }
    pub(crate) fn rigid(&self, index: usize) -> bool {
        self.rigid.as_ref().is_some_and(|flags| flags[index] != 0)
    }
    pub(crate) fn should_skip(&self, index: usize) -> bool {
        !self.alive(index) || self.rigid.as_ref().is_some_and(|flags| flags[index] != 0)
    }
}

pub(crate) fn mask_labels(
    objects: &PhysObj,
    selection: ParticleSelection,
) -> [Option<&'static str>; 2] {
    [
        (!selection.includes_dead() && objects.core.contains(ATTR_ALIVE)).then_some(ATTR_ALIVE),
        objects.core.contains(ATTR_RIGID).then_some(ATTR_RIGID),
    ]
}

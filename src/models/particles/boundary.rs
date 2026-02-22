/*!
General boundary-condition tools for particle models.
*/

use crate::engines::soa::phys_obj::{AttrsError, PhysObj};
use crate::models::particles::attrs::{ATTR_ALIVE, ATTR_R, ATTR_V};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryError {
    InvalidBounds {
        axis: usize,
        min: f64,
        max: f64,
    },
    Attrs(AttrsError),
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
}

impl From<AttrsError> for BoundaryError {
    #[inline]
    fn from(value: AttrsError) -> Self {
        Self::Attrs(value)
    }
}

pub trait Boundary: Sync {
    /// - Purpose: Applies this boundary condition directly to the canonical particle attributes in `PhysObj`.
    /// - Parameters:
    ///   - `objects` (`&mut PhysObj`): SoA object storage containing position/velocity attributes to update in-place.
    fn apply(&self, objects: &mut PhysObj) -> Result<(), BoundaryError>;
}

fn validate_bounds(min: &[f64], max: &[f64]) -> Result<(), BoundaryError> {
    if min.len() != max.len() {
        return Err(BoundaryError::InvalidAttrShape {
            label: "bounds",
            expected_dim: min.len(),
            got_dim: max.len(),
        });
    }
    for d in 0..min.len() {
        if !min[d].is_finite() || !max[d].is_finite() || max[d] <= min[d] {
            return Err(BoundaryError::InvalidBounds {
                axis: d,
                min: min[d],
                max: max[d],
            });
        }
    }
    Ok(())
}

#[inline]
fn shape_and_alive(objects: &PhysObj) -> Result<(usize, usize, Option<Vec<bool>>), BoundaryError> {
    let (dim, n) = {
        let r = objects.core.get::<f64>(ATTR_R)?;
        (r.dim(), r.num_vectors())
    };

    {
        let v = objects.core.get::<f64>(ATTR_V)?;
        if v.dim() != dim {
            return Err(BoundaryError::InvalidAttrShape {
                label: ATTR_V,
                expected_dim: dim,
                got_dim: v.dim(),
            });
        }
        if v.num_vectors() != n {
            return Err(BoundaryError::InconsistentParticleCount {
                label: ATTR_V,
                expected: n,
                got: v.num_vectors(),
            });
        }
    }

    let alive_flags = if !objects.core.contains(ATTR_ALIVE) {
        None
    } else {
        let alive = objects.core.get::<f64>(ATTR_ALIVE)?;
        if alive.dim() != 1 {
            return Err(BoundaryError::InvalidAttrShape {
                label: ATTR_ALIVE,
                expected_dim: 1,
                got_dim: alive.dim(),
            });
        }
        if alive.num_vectors() != n {
            return Err(BoundaryError::InconsistentParticleCount {
                label: ATTR_ALIVE,
                expected: n,
                got: alive.num_vectors(),
            });
        }

        let flags = alive
            .as_tensor()
            .data
            .par_iter()
            .map(|&x| x > 0.0)
            .collect::<Vec<bool>>();
        Some(flags)
    };

    Ok((dim, n, alive_flags))
}

#[derive(Debug, Clone)]
pub struct PeriodicBox {
    min: Vec<f64>,
    max: Vec<f64>,
}

impl PeriodicBox {
    /// - Purpose: Constructs a periodic box from per-axis lower/upper bounds.
    /// - Parameters:
    ///   - `min` (`&[f64]`): Per-axis inclusive lower bounds provided by caller.
    ///   - `max` (`&[f64]`): Per-axis exclusive upper bounds provided by caller.
    pub fn new(min: &[f64], max: &[f64]) -> Result<Self, BoundaryError> {
        validate_bounds(min, max)?;
        Ok(Self {
            min: min.to_vec(),
            max: max.to_vec(),
        })
    }
}

impl Boundary for PeriodicBox {
    fn apply(&self, objects: &mut PhysObj) -> Result<(), BoundaryError> {
        let (dim, _n, alive_flags) = shape_and_alive(objects)?;

        let r = objects.core.get_mut::<f64>(ATTR_R)?;
        r.as_tensor_mut()
            .data
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(i, row)| {
                if alive_flags.as_ref().is_some_and(|flags| !flags[i]) {
                    return;
                }

                for (d, x) in row.iter_mut().enumerate() {
                    if !x.is_finite() {
                        continue;
                    }
                    let lo = self.min[d];
                    let hi = self.max[d];
                    let w = hi - lo;
                    *x = lo + (*x - lo).rem_euclid(w);
                }
            });

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ClampBox {
    min: Vec<f64>,
    max: Vec<f64>,
}

impl ClampBox {
    /// - Purpose: Constructs a clamping box from per-axis lower/upper bounds.
    /// - Parameters:
    ///   - `min` (`&[f64]`): Per-axis lower clamp values provided by caller.
    ///   - `max` (`&[f64]`): Per-axis upper clamp values provided by caller.
    pub fn new(min: &[f64], max: &[f64]) -> Result<Self, BoundaryError> {
        validate_bounds(min, max)?;
        Ok(Self {
            min: min.to_vec(),
            max: max.to_vec(),
        })
    }
}

impl Boundary for ClampBox {
    fn apply(&self, objects: &mut PhysObj) -> Result<(), BoundaryError> {
        let (dim, _n, alive_flags) = shape_and_alive(objects)?;

        let r = objects.core.get_mut::<f64>(ATTR_R)?;
        r.as_tensor_mut()
            .data
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(i, row)| {
                if alive_flags.as_ref().is_some_and(|flags| !flags[i]) {
                    return;
                }

                for (d, x) in row.iter_mut().enumerate() {
                    *x = x.clamp(self.min[d], self.max[d]);
                }
            });

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ReflectBox {
    min: Vec<f64>,
    max: Vec<f64>,
}

impl ReflectBox {
    /// - Purpose: Constructs a reflecting box from per-axis lower/upper bounds.
    /// - Parameters:
    ///   - `min` (`&[f64]`): Per-axis lower bounds used for reflection planes.
    ///   - `max` (`&[f64]`): Per-axis upper bounds used for reflection planes.
    pub fn new(min: &[f64], max: &[f64]) -> Result<Self, BoundaryError> {
        validate_bounds(min, max)?;
        Ok(Self {
            min: min.to_vec(),
            max: max.to_vec(),
        })
    }
}

impl Boundary for ReflectBox {
    fn apply(&self, objects: &mut PhysObj) -> Result<(), BoundaryError> {
        let (dim, n, alive_flags) = shape_and_alive(objects)?;
        let mut flip_mask: Vec<u8> = vec![0; n * dim];

        {
            let r = objects.core.get_mut::<f64>(ATTR_R)?;
            r.as_tensor_mut()
                .data
                .par_chunks_mut(dim)
                .zip(flip_mask.par_chunks_mut(dim))
                .enumerate()
                .for_each(|(i, (r_row, mask_row))| {
                    if alive_flags.as_ref().is_some_and(|flags| !flags[i]) {
                        return;
                    }

                    for d in 0..dim {
                        let x = r_row[d];
                        if !x.is_finite() {
                            continue;
                        }

                        let lo = self.min[d];
                        let hi = self.max[d];
                        if !(x < lo || x > hi) {
                            continue;
                        }

                        let w = hi - lo;
                        let y = (x - lo).rem_euclid(2.0 * w);
                        r_row[d] = if y <= w { lo + y } else { hi - (y - w) };

                        let flips = if x < lo {
                            ((lo - x) / w).ceil() as i64
                        } else {
                            ((x - hi) / w).ceil() as i64
                        };
                        if flips & 1 == 1 {
                            mask_row[d] = 1;
                        }
                    }
                });
        }

        {
            let v = objects.core.get_mut::<f64>(ATTR_V)?;
            v.as_tensor_mut()
                .data
                .par_chunks_mut(dim)
                .zip(flip_mask.par_chunks(dim))
                .enumerate()
                .for_each(|(i, (v_row, mask_row))| {
                    if alive_flags.as_ref().is_some_and(|flags| !flags[i]) {
                        return;
                    }

                    for d in 0..dim {
                        if mask_row[d] == 1 {
                            v_row[d] = -v_row[d];
                        }
                    }
                });
        }

        Ok(())
    }
}

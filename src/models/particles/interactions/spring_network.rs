//! Validated undirected spring interactions for canonical particle state.

use core::fmt;
use std::collections::HashSet;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::engines::soa::interaction::InteractionOrder;
use crate::engines::soa::phys_obj::PhysObj;
use crate::engines::soa::{Interaction, InteractionError};
use crate::models::laws::{Spring, SpringLawError};
use crate::models::particles::attrs::{ATTR_A, ATTR_R, ParticleSelection};
use crate::models::particles::state::{
    ParticleStateError, gather_inverse_mass, gather_masks, validate_vector_attr_f64,
};

/// Errors returned by spring-network operations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SpringNetworkError {
    State(ParticleStateError),
    Law(SpringLawError),
    SelfPair {
        particle: usize,
    },
    ParticleCountOverflow {
        particle: usize,
    },
    DuplicatePair {
        pair: (usize, usize),
    },
    EndpointOutOfBounds {
        particle: usize,
        particle_count: usize,
    },
    InvalidInverseMass {
        particle: usize,
        value: f64,
    },
    InternalInvariant {
        message: String,
    },
}

impl From<ParticleStateError> for SpringNetworkError {
    fn from(error: ParticleStateError) -> Self {
        Self::State(error)
    }
}

impl From<SpringLawError> for SpringNetworkError {
    fn from(error: SpringLawError) -> Self {
        Self::Law(error)
    }
}

impl fmt::Display for SpringNetworkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(error) => write!(formatter, "invalid particle state: {error}"),
            Self::Law(error) => write!(formatter, "invalid spring law: {error}"),
            Self::SelfPair { particle } => {
                write!(formatter, "spring pair cannot repeat particle {particle}")
            }
            Self::ParticleCountOverflow { particle } => write!(
                formatter,
                "spring endpoint {particle} cannot be represented as a particle-count bound"
            ),
            Self::DuplicatePair { pair } => {
                write!(formatter, "spring batch contains duplicate pair {pair:?}")
            }
            Self::EndpointOutOfBounds {
                particle,
                particle_count,
            } => write!(
                formatter,
                "spring endpoint {particle} is outside particle state with {particle_count} particles"
            ),
            Self::InvalidInverseMass { particle, value } => write!(
                formatter,
                "inverse mass at particle {particle} must be finite and non-negative; got {value}"
            ),
            Self::InternalInvariant { message } => {
                write!(formatter, "spring-network invariant failed: {message}")
            }
        }
    }
}

impl std::error::Error for SpringNetworkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(error) => Some(error),
            Self::Law(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SpringRecord {
    pair: (usize, usize),
    law: Spring,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SpringNetworkDocument {
    springs: Vec<SpringRecord>,
}

/// Sparse undirected collection keyed by particle pairs.
#[derive(Debug, Clone)]
pub struct SpringNetwork {
    springs: Interaction<Spring>,
}

impl Default for SpringNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl Serialize for SpringNetwork {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SpringNetworkDocument {
            springs: self
                .iter()
                .map(|(pair, law)| SpringRecord { pair, law: *law })
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SpringNetwork {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let document = SpringNetworkDocument::deserialize(deserializer)?;
        let mut seen = HashSet::with_capacity(document.springs.len());
        let mut network = Self::with_capacity(document.springs.len());
        for record in document.springs {
            let pair = canonical_pair(record.pair);
            if !seen.insert(pair) {
                return Err(serde::de::Error::custom(
                    SpringNetworkError::DuplicatePair { pair },
                ));
            }
            network
                .insert(pair, record.law)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(network)
    }
}

impl SpringNetwork {
    pub fn new() -> Self {
        Self {
            springs: Interaction::new(0, InteractionOrder::Unordered),
        }
    }

    /// Reserves edge storage without imposing a particle count.
    pub fn with_capacity(edge_capacity: usize) -> Self {
        let mut network = Self::new();
        network.springs.reserve(edge_capacity);
        network
    }

    pub fn len(&self) -> usize {
        self.springs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.springs.is_empty()
    }

    /// Smallest particle count capable of containing every current endpoint.
    pub fn minimum_particle_count(&self) -> usize {
        self.iter()
            .map(|((i, j), _)| i.max(j))
            .max()
            .map_or(0, |particle| particle + 1)
    }

    /// Inserts or replaces one pair after validating the complete request.
    pub fn insert(
        &mut self,
        pair: (usize, usize),
        law: Spring,
    ) -> Result<Option<Spring>, SpringNetworkError> {
        law.validate()?;
        let pair = validate_pair(pair)?;
        let needed = pair
            .1
            .checked_add(1)
            .ok_or(SpringNetworkError::ParticleCountOverflow { particle: pair.1 })?;
        let previous = self.get(pair).copied();
        self.ensure_particle_bound(needed)?;
        self.springs
            .set_pair(pair.0, pair.1, law)
            .map_err(invariant)?;
        Ok(previous)
    }

    /// Applies a batch atomically. Duplicate pairs in the batch are rejected.
    pub fn insert_many(
        &mut self,
        entries: impl IntoIterator<Item = ((usize, usize), Spring)>,
    ) -> Result<(), SpringNetworkError> {
        let mut checked = Vec::new();
        let mut seen = HashSet::new();
        let mut needed = self.springs.n_objects();
        for (pair, law) in entries {
            law.validate()?;
            let pair = validate_pair(pair)?;
            if !seen.insert(pair) {
                return Err(SpringNetworkError::DuplicatePair { pair });
            }
            needed = needed.max(
                pair.1
                    .checked_add(1)
                    .ok_or(SpringNetworkError::ParticleCountOverflow { particle: pair.1 })?,
            );
            checked.push((pair, law));
        }

        let mut replacement = self.clone();
        replacement.ensure_particle_bound(needed)?;
        replacement.springs.reserve(checked.len());
        for (pair, law) in checked {
            replacement
                .springs
                .set_pair(pair.0, pair.1, law)
                .map_err(invariant)?;
        }
        *self = replacement;
        Ok(())
    }

    pub fn get(&self, pair: (usize, usize)) -> Option<&Spring> {
        let pair = canonical_pair(pair);
        if pair.1 >= self.springs.n_objects() || pair.0 == pair.1 {
            return None;
        }
        self.springs.get_pair(pair.0, pair.1).ok().flatten()
    }

    pub fn remove(&mut self, pair: (usize, usize)) -> Result<Option<Spring>, SpringNetworkError> {
        let pair = canonical_pair(pair);
        if pair.1 >= self.springs.n_objects() || pair.0 == pair.1 {
            return Ok(None);
        }
        Ok(self
            .springs
            .remove_pair(pair.0, pair.1)
            .map_err(invariant)?
            .map(|(_, law)| law))
    }

    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), &Spring)> + '_ {
        self.springs.iter().filter_map(|(_, nodes, law)| {
            (nodes.nodes.len() == 2).then(|| ((nodes.nodes[0], nodes.nodes[1]), law))
        })
    }

    pub fn clear(&mut self) {
        self.springs.clear();
    }

    /// Adds Hooke-law acceleration contributions to canonical particle state.
    pub fn apply(
        &self,
        objects: &mut PhysObj,
        selection: ParticleSelection,
    ) -> Result<(), SpringNetworkError> {
        let (dim, particle_count, positions) = {
            let positions = objects
                .core
                .get::<f64>(ATTR_R)
                .map_err(ParticleStateError::from)?;
            (
                positions.dim(),
                positions.num_vectors(),
                positions.as_tensor().data.clone(),
            )
        };
        validate_vector_attr_f64(objects, ATTR_A, dim, particle_count)?;
        let inverse_mass = gather_inverse_mass(objects, particle_count)?;
        let masks = gather_masks(objects, particle_count, selection)?;
        for (particle, &value) in inverse_mass.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(SpringNetworkError::InvalidInverseMass { particle, value });
            }
        }

        let mut acceleration = vec![0.0; particle_count * dim];
        for ((i, j), law) in self.iter() {
            for endpoint in [i, j] {
                if endpoint >= particle_count {
                    return Err(SpringNetworkError::EndpointOutOfBounds {
                        particle: endpoint,
                        particle_count,
                    });
                }
            }
            if !masks.is_included(selection, i) || !masks.is_included(selection, j) {
                continue;
            }
            let i_base = i * dim;
            let j_base = j * dim;
            let norm_squared = (0..dim)
                .map(|component| {
                    let delta = positions[i_base + component] - positions[j_base + component];
                    delta * delta
                })
                .sum::<f64>();
            if !norm_squared.is_finite() || norm_squared <= f64::EPSILON {
                continue;
            }
            let norm = norm_squared.sqrt();
            if law
                .cutoff()
                .is_some_and(|(minimum, maximum)| norm < minimum || norm > maximum)
            {
                continue;
            }
            let scale = -law.spring_constant() * (norm - law.rest_length()) / norm;
            let i_rigid = masks.rigid.as_ref().is_some_and(|flags| flags[i]);
            let j_rigid = masks.rigid.as_ref().is_some_and(|flags| flags[j]);
            for component in 0..dim {
                let force = (positions[i_base + component] - positions[j_base + component]) * scale;
                if !i_rigid {
                    acceleration[i_base + component] += force * inverse_mass[i];
                }
                if !j_rigid {
                    acceleration[j_base + component] -= force * inverse_mass[j];
                }
            }
        }

        let output = objects
            .core
            .get_mut::<f64>(ATTR_A)
            .map_err(ParticleStateError::from)?;
        for (destination, contribution) in output.as_tensor_mut().data.iter_mut().zip(acceleration)
        {
            *destination += contribution;
        }
        Ok(())
    }

    fn ensure_particle_bound(&mut self, needed: usize) -> Result<(), SpringNetworkError> {
        if needed > self.springs.n_objects() {
            self.springs.set_n_objects(needed).map_err(invariant)?;
        }
        Ok(())
    }
}

fn canonical_pair((i, j): (usize, usize)) -> (usize, usize) {
    if i <= j { (i, j) } else { (j, i) }
}

fn validate_pair(pair: (usize, usize)) -> Result<(usize, usize), SpringNetworkError> {
    let pair = canonical_pair(pair);
    if pair.0 == pair.1 {
        return Err(SpringNetworkError::SelfPair { particle: pair.0 });
    }
    Ok(pair)
}

fn invariant(error: InteractionError) -> SpringNetworkError {
    SpringNetworkError::InternalInvariant {
        message: error.to_string(),
    }
}

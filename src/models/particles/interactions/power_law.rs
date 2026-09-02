//! Validated undirected power-law interactions for canonical particle state.

use core::fmt;
use std::collections::HashSet;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::engines::soa::interaction::InteractionOrder;
use crate::engines::soa::phys_obj::PhysObj;
use crate::engines::soa::{Interaction, InteractionError};
use crate::models::laws::{PowerLawDecay, PowerLawError};
use crate::models::particles::attrs::{ATTR_A, ATTR_R, ParticleSelection};
use crate::models::particles::state::{
    ParticleStateError, gather_inverse_mass, gather_masks, validate_vector_attr_f64,
};

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PowerLawNetworkError {
    State(ParticleStateError),
    Law(PowerLawError),
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

impl From<ParticleStateError> for PowerLawNetworkError {
    fn from(error: ParticleStateError) -> Self {
        Self::State(error)
    }
}

impl From<PowerLawError> for PowerLawNetworkError {
    fn from(error: PowerLawError) -> Self {
        Self::Law(error)
    }
}

impl fmt::Display for PowerLawNetworkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(error) => write!(formatter, "invalid particle state: {error}"),
            Self::Law(error) => write!(formatter, "invalid power-law: {error}"),
            Self::SelfPair { particle } => write!(
                formatter,
                "power-law pair cannot repeat particle {particle}"
            ),
            Self::ParticleCountOverflow { particle } => write!(
                formatter,
                "power-law endpoint {particle} cannot be represented as a particle-count bound"
            ),
            Self::DuplicatePair { pair } => {
                write!(
                    formatter,
                    "power-law batch contains duplicate pair {pair:?}"
                )
            }
            Self::EndpointOutOfBounds {
                particle,
                particle_count,
            } => write!(
                formatter,
                "power-law endpoint {particle} is outside particle state with {particle_count} particles"
            ),
            Self::InvalidInverseMass { particle, value } => write!(
                formatter,
                "inverse mass at particle {particle} must be finite and non-negative; got {value}"
            ),
            Self::InternalInvariant { message } => {
                write!(formatter, "power-law-network invariant failed: {message}")
            }
        }
    }
}

impl std::error::Error for PowerLawNetworkError {
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
struct PowerLawRecord {
    pair: (usize, usize),
    law: PowerLawDecay,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PowerLawNetworkDocument {
    interactions: Vec<PowerLawRecord>,
}

#[derive(Debug, Clone)]
pub struct PowerLawNetwork {
    interactions: Interaction<PowerLawDecay>,
}

impl Default for PowerLawNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl Serialize for PowerLawNetwork {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PowerLawNetworkDocument {
            interactions: self
                .iter()
                .map(|(pair, law)| PowerLawRecord { pair, law: *law })
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PowerLawNetwork {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let document = PowerLawNetworkDocument::deserialize(deserializer)?;
        let mut seen = HashSet::with_capacity(document.interactions.len());
        let mut network = Self::with_capacity(document.interactions.len());
        for record in document.interactions {
            let pair = canonical_pair(record.pair);
            if !seen.insert(pair) {
                return Err(serde::de::Error::custom(
                    PowerLawNetworkError::DuplicatePair { pair },
                ));
            }
            network
                .insert(pair, record.law)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(network)
    }
}

impl PowerLawNetwork {
    pub fn new() -> Self {
        Self {
            interactions: Interaction::new(0, InteractionOrder::Unordered),
        }
    }

    pub fn with_capacity(edge_capacity: usize) -> Self {
        let mut network = Self::new();
        network.interactions.reserve(edge_capacity);
        network
    }

    pub fn len(&self) -> usize {
        self.interactions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.interactions.is_empty()
    }

    pub fn minimum_particle_count(&self) -> usize {
        self.iter()
            .map(|((i, j), _)| i.max(j))
            .max()
            .map_or(0, |particle| particle + 1)
    }

    pub fn insert(
        &mut self,
        pair: (usize, usize),
        law: PowerLawDecay,
    ) -> Result<Option<PowerLawDecay>, PowerLawNetworkError> {
        law.validate()?;
        let pair = validate_pair(pair)?;
        let needed = pair
            .1
            .checked_add(1)
            .ok_or(PowerLawNetworkError::ParticleCountOverflow { particle: pair.1 })?;
        let previous = self.get(pair).copied();
        self.ensure_particle_bound(needed)?;
        self.interactions
            .set_pair(pair.0, pair.1, law)
            .map_err(invariant)?;
        Ok(previous)
    }

    pub fn insert_many(
        &mut self,
        entries: impl IntoIterator<Item = ((usize, usize), PowerLawDecay)>,
    ) -> Result<(), PowerLawNetworkError> {
        let mut checked = Vec::new();
        let mut seen = HashSet::new();
        let mut needed = self.interactions.n_objects();
        for (pair, law) in entries {
            law.validate()?;
            let pair = validate_pair(pair)?;
            if !seen.insert(pair) {
                return Err(PowerLawNetworkError::DuplicatePair { pair });
            }
            needed = needed.max(
                pair.1
                    .checked_add(1)
                    .ok_or(PowerLawNetworkError::ParticleCountOverflow { particle: pair.1 })?,
            );
            checked.push((pair, law));
        }

        let mut replacement = self.clone();
        replacement.ensure_particle_bound(needed)?;
        replacement.interactions.reserve(checked.len());
        for (pair, law) in checked {
            replacement
                .interactions
                .set_pair(pair.0, pair.1, law)
                .map_err(invariant)?;
        }
        *self = replacement;
        Ok(())
    }

    /// Inserts one law on every unordered pair in `0..particle_count`.
    pub fn insert_all_to_all(
        &mut self,
        particle_count: usize,
        law: PowerLawDecay,
    ) -> Result<(), PowerLawNetworkError> {
        law.validate()?;
        let pair_count = particle_count
            .checked_mul(particle_count.saturating_sub(1))
            .and_then(|count| count.checked_div(2))
            .ok_or(PowerLawNetworkError::ParticleCountOverflow {
                particle: particle_count,
            })?;
        let entries =
            (0..particle_count).flat_map(|i| ((i + 1)..particle_count).map(move |j| ((i, j), law)));
        let mut replacement = self.clone();
        replacement.interactions.reserve(pair_count);
        replacement.insert_many(entries)?;
        *self = replacement;
        Ok(())
    }

    pub fn get(&self, pair: (usize, usize)) -> Option<&PowerLawDecay> {
        let pair = canonical_pair(pair);
        if pair.1 >= self.interactions.n_objects() || pair.0 == pair.1 {
            return None;
        }
        self.interactions.get_pair(pair.0, pair.1).ok().flatten()
    }

    pub fn remove(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<PowerLawDecay>, PowerLawNetworkError> {
        let pair = canonical_pair(pair);
        if pair.1 >= self.interactions.n_objects() || pair.0 == pair.1 {
            return Ok(None);
        }
        Ok(self
            .interactions
            .remove_pair(pair.0, pair.1)
            .map_err(invariant)?
            .map(|(_, law)| law))
    }

    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), &PowerLawDecay)> + '_ {
        self.interactions.iter().filter_map(|(_, nodes, law)| {
            (nodes.nodes.len() == 2).then(|| ((nodes.nodes[0], nodes.nodes[1]), law))
        })
    }

    pub fn clear(&mut self) {
        self.interactions.clear();
    }

    pub fn apply(
        &self,
        objects: &mut PhysObj,
        selection: ParticleSelection,
    ) -> Result<(), PowerLawNetworkError> {
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
                return Err(PowerLawNetworkError::InvalidInverseMass { particle, value });
            }
        }

        let mut acceleration = vec![0.0; particle_count * dim];
        for ((i, j), law) in self.iter() {
            for endpoint in [i, j] {
                if endpoint >= particle_count {
                    return Err(PowerLawNetworkError::EndpointOutOfBounds {
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
                .range()
                .is_some_and(|(minimum, maximum)| norm < minimum || norm > maximum)
            {
                continue;
            }
            let scale = law.strength() * norm.powf(law.exponent() - 1.0);
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

    fn ensure_particle_bound(&mut self, needed: usize) -> Result<(), PowerLawNetworkError> {
        if needed > self.interactions.n_objects() {
            self.interactions.set_n_objects(needed).map_err(invariant)?;
        }
        Ok(())
    }
}

fn canonical_pair((i, j): (usize, usize)) -> (usize, usize) {
    if i <= j { (i, j) } else { (j, i) }
}

fn validate_pair(pair: (usize, usize)) -> Result<(usize, usize), PowerLawNetworkError> {
    let pair = canonical_pair(pair);
    if pair.0 == pair.1 {
        return Err(PowerLawNetworkError::SelfPair { particle: pair.0 });
    }
    Ok(pair)
}

fn invariant(error: InteractionError) -> PowerLawNetworkError {
    PowerLawNetworkError::InternalInvariant {
        message: error.to_string(),
    }
}

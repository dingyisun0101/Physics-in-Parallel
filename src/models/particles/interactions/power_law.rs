/*!
Pairwise power-law interaction parameter storage for particle models.

Purpose:
`PowerLawNetwork` stores unordered particle pairs with one `PowerLawDecay`
payload on each pair. The struct is currently a validated storage layer; actual
force or rate application belongs in downstream model code that knows the
physical convention being used.
*/

use crate::engines::soa::interaction::InteractionOrder;
use crate::engines::soa::{Interaction, InteractionError, InteractionId};

/// Optional active distance interval `(min, max)`.
pub type PowerLawRange = (f64, f64);

/// Per-pair power-law payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerLawDecay {
    /// Strength constant.
    pub k: f64,
    /// Power exponent.
    pub alpha: f64,
    /// Optional active distance interval `(min, max)`.
    pub range: Option<PowerLawRange>,
}

impl PowerLawDecay {
    /// Builds a validated power-law payload.
    pub fn new(
        k: f64,
        alpha: f64,
        range: Option<PowerLawRange>,
    ) -> Result<Self, PowerLawNetworkError> {
        let payload = Self { k, alpha, range };
        payload.validate()?;
        Ok(payload)
    }

    /// Validates power-law parameters.
    pub fn validate(&self) -> Result<(), PowerLawNetworkError> {
        if !self.k.is_finite() {
            return Err(PowerLawNetworkError::InvalidStrength { k: self.k });
        }
        if !self.alpha.is_finite() {
            return Err(PowerLawNetworkError::InvalidExponent { alpha: self.alpha });
        }
        if let Some((min, max)) = self.range {
            if !min.is_finite() || !max.is_finite() || min < 0.0 || max < min {
                return Err(PowerLawNetworkError::InvalidRange { min, max });
            }
        }
        Ok(())
    }
}

/// Errors returned by power-law network operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PowerLawNetworkError {
    /// Lower-level interaction storage error.
    Interaction(InteractionError),
    /// Strength constant is not finite.
    InvalidStrength {
        /// Invalid strength value.
        k: f64,
    },
    /// Power exponent is not finite.
    InvalidExponent {
        /// Invalid exponent value.
        alpha: f64,
    },
    /// Active range is not finite, negative, or ordered incorrectly.
    InvalidRange {
        /// Lower active distance.
        min: f64,
        /// Upper active distance.
        max: f64,
    },
}

impl From<InteractionError> for PowerLawNetworkError {
    fn from(value: InteractionError) -> Self {
        Self::Interaction(value)
    }
}

/// Undirected network of pairwise power-law interaction parameters.
#[derive(Debug, Clone)]
pub struct PowerLawNetwork {
    interactions: Interaction<PowerLawDecay>,
}

impl Default for PowerLawNetwork {
    fn default() -> Self {
        Self::empty()
    }
}

impl PowerLawNetwork {
    /// Creates an empty power-law network.
    pub fn empty() -> Self {
        Self {
            interactions: Interaction::new(0, InteractionOrder::Unordered),
        }
    }

    /// Number of active pair interactions.
    pub fn len(&self) -> usize {
        self.interactions.len()
    }

    /// Returns true if no interactions exist.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds or overwrites one pair interaction payload from constants.
    pub fn add_power_law(
        &mut self,
        pair: (usize, usize),
        k: f64,
        alpha: f64,
        range: Option<PowerLawRange>,
    ) -> Result<InteractionId, PowerLawNetworkError> {
        self.add_payload(pair, PowerLawDecay::new(k, alpha, range)?)
    }

    /// Adds or overwrites one pair interaction payload.
    pub fn add_payload(
        &mut self,
        pair: (usize, usize),
        payload: PowerLawDecay,
    ) -> Result<InteractionId, PowerLawNetworkError> {
        payload.validate()?;
        self.ensure_n_objects_for(pair);
        Ok(self.interactions.set_pair(pair.0, pair.1, payload)?)
    }

    /// Removes one pair interaction payload.
    pub fn remove_power_law(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<PowerLawDecay>, PowerLawNetworkError> {
        if pair.0.max(pair.1) >= self.interactions.topology().n_objects() {
            return Ok(None);
        }
        Ok(self
            .interactions
            .remove_pair(pair.0, pair.1)?
            .map(|(_, payload)| payload))
    }

    /// Returns immutable payload for one pair.
    pub fn get_power_law(
        &self,
        pair: (usize, usize),
    ) -> Result<Option<&PowerLawDecay>, PowerLawNetworkError> {
        if pair.0.max(pair.1) >= self.interactions.topology().n_objects() {
            return Ok(None);
        }
        Ok(self.interactions.get_pair(pair.0, pair.1)?)
    }

    /// Returns mutable payload for one pair.
    pub fn get_power_law_mut(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<&mut PowerLawDecay>, PowerLawNetworkError> {
        if pair.0.max(pair.1) >= self.interactions.topology().n_objects() {
            return Ok(None);
        }
        Ok(self.interactions.get_pair_mut(pair.0, pair.1)?)
    }

    /// Clears all interactions while preserving capacity.
    pub fn clear(&mut self) {
        self.interactions.clear();
    }

    /// Read-only access to the wrapped interaction backend.
    pub fn interaction(&self) -> &Interaction<PowerLawDecay> {
        &self.interactions
    }

    /// Mutable access to the wrapped interaction backend.
    pub fn interaction_mut(&mut self) -> &mut Interaction<PowerLawDecay> {
        &mut self.interactions
    }

    fn ensure_n_objects_for(&mut self, pair: (usize, usize)) {
        let needed = pair.0.max(pair.1).saturating_add(1);
        if needed > self.interactions.topology().n_objects() {
            self.interactions
                .set_n_objects(needed)
                .expect("growing power-law interaction object bound should not invalidate entries");
        }
    }
}

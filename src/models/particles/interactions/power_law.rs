/*!
Pairwise power-law interaction parameter storage for particle models.

Purpose:
`PowerLawNetwork` stores unordered particle pairs with one reusable
`models::laws::PowerLawDecay` payload on each pair. The struct is currently a
validated storage layer; actual force or rate application belongs in downstream
model code that knows the physical convention being used.
*/

use crate::engines::soa::interaction::InteractionOrder;
use crate::engines::soa::{Interaction, InteractionError, InteractionId};
use crate::models::laws::{PowerLawDecay, PowerLawError, PowerLawRange};

/// Errors returned by power-law network operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PowerLawNetworkError {
    /// Lower-level interaction storage error.
    Interaction(InteractionError),
    /// Lower-level power-law validation error.
    Law(PowerLawError),
}

impl From<InteractionError> for PowerLawNetworkError {
    fn from(value: InteractionError) -> Self {
        Self::Interaction(value)
    }
}

impl From<PowerLawError> for PowerLawNetworkError {
    fn from(value: PowerLawError) -> Self {
        Self::Law(value)
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

/*!
Power-law interaction wrapper for particle models.
*/

use crate::engines::soa::interaction::DirectionMode;
use crate::engines::soa::{EdgeId, Interaction, InteractionError};

/// Optional active range for this interaction, encoded as `(max, min)`.
pub type PowerLawRange = (f64, f64);

/// Per-edge power-law interaction payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerLawDecay {
    /// Strength constant.
    pub k: f64,
    /// Power exponent.
    pub alpha: f64,
    /// Optional active range `(max, min)`.
    pub range: Option<PowerLawRange>,
}

impl PowerLawDecay {
    /// Creates a power-law payload from constants.
    pub fn new(k: f64, alpha: f64, range: Option<PowerLawRange>) -> Self {
        Self { k, alpha, range }
    }
}

/// Undirected network of pairwise power-law interactions.
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
            interactions: Interaction::new(0, DirectionMode::Undirected),
        }
    }

    /// Number of active pair interactions.
    pub fn len(&self) -> usize {
        self.interactions.topology().active_count()
    }

    /// Returns true if no interactions exist.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds or overwrites one pair interaction payload.
    pub fn add(
        &mut self,
        pair: (usize, usize),
        k: f64,
        alpha: f64,
        range: Option<PowerLawRange>,
    ) -> Result<EdgeId, InteractionError> {
        self.add_power_law(pair, k, alpha, range)
    }

    /// Deletes one pair interaction payload.
    pub fn delete(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<PowerLawDecay>, InteractionError> {
        self.delete_power_law(pair)
    }

    /// Adds or overwrites one pair interaction payload.
    pub fn add_power_law(
        &mut self,
        pair: (usize, usize),
        k: f64,
        alpha: f64,
        range: Option<PowerLawRange>,
    ) -> Result<EdgeId, InteractionError> {
        self.add_payload(pair, PowerLawDecay::new(k, alpha, range))
    }

    /// Adds or overwrites one pair interaction payload.
    pub fn add_payload(
        &mut self,
        pair: (usize, usize),
        payload: PowerLawDecay,
    ) -> Result<EdgeId, InteractionError> {
        self.ensure_n_objects_for(pair);
        self.interactions.insert(&[pair.0, pair.1], payload)
    }

    /// Deletes one pair interaction payload.
    pub fn delete_power_law(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<PowerLawDecay>, InteractionError> {
        if pair.0.max(pair.1) >= self.interactions.topology().n_objects() {
            return Ok(None);
        }
        Ok(self.interactions.remove(&[pair.0, pair.1])?.map(|(_, payload)| payload))
    }

    /// Alias for `delete_power_law`.
    pub fn remove_power_law(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<PowerLawDecay>, InteractionError> {
        self.delete_power_law(pair)
    }

    /// Returns immutable payload for one pair.
    pub fn get_power_law(
        &self,
        pair: (usize, usize),
    ) -> Result<Option<&PowerLawDecay>, InteractionError> {
        if pair.0.max(pair.1) >= self.interactions.topology().n_objects() {
            return Ok(None);
        }
        self.interactions.get(&[pair.0, pair.1])
    }

    /// Returns mutable payload for one pair.
    pub fn get_power_law_mut(
        &mut self,
        pair: (usize, usize),
    ) -> Result<Option<&mut PowerLawDecay>, InteractionError> {
        if pair.0.max(pair.1) >= self.interactions.topology().n_objects() {
            return Ok(None);
        }
        self.interactions.get_mut(&[pair.0, pair.1])
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
            self.interactions.topology_mut().set_n_objects(needed);
        }
    }
}

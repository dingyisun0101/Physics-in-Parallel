//! PiP-owned, fully resolved randomness.
//!
//! Every public stochastic API accepts [`ResolvedRng`]. PiP never chooses a
//! hidden seed or method on behalf of a caller. Applications may construct a
//! reproducible value with [`ResolvedRng::new`] or explicitly request host
//! entropy with [`ResolvedRng::from_entropy`].

use std::fmt;

use serde::{Deserialize, Serialize};

/// Random methods implemented by PiP stochastic components.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RngMethod {
    IndexedSplitMix64,
    Pcg64,
    Pcg64Mcg,
    SmallRng,
    ChaCha8,
    ChaCha12,
    ChaCha20,
}

impl RngMethod {
    /// Stable method identifier for scientific provenance.
    pub const fn name(self) -> &'static str {
        match self {
            Self::IndexedSplitMix64 => "splitmix64_indexed",
            Self::Pcg64 => "pcg64",
            Self::Pcg64Mcg => "pcg64_mcg",
            Self::SmallRng => "small_rng",
            Self::ChaCha8 => "chacha8",
            Self::ChaCha12 => "chacha12",
            Self::ChaCha20 => "chacha20",
        }
    }

    /// Sequence-affecting implementation version.
    pub const fn version(self) -> &'static str {
        match self {
            Self::IndexedSplitMix64 => "1",
            Self::Pcg64 | Self::Pcg64Mcg => "rand_pcg-0.10.2",
            Self::SmallRng => "rand-0.10.2",
            Self::ChaCha8 | Self::ChaCha12 | Self::ChaCha20 => "rand_chacha-0.10",
        }
    }

    /// Stable encoding used for the root seed.
    pub const fn seed_encoding(self) -> &'static str {
        "u64_decimal"
    }

    /// Parses a user-facing method name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "indexed" | "splitmix64" | "splitmix64_indexed" => Some(Self::IndexedSplitMix64),
            "pcg64" => Some(Self::Pcg64),
            "pcg64mcg" | "pcg64_mcg" | "pcg64-fast" | "pcg64fast" => Some(Self::Pcg64Mcg),
            "small" | "small_rng" | "smallrng" => Some(Self::SmallRng),
            "chacha8" | "chacha8rng" => Some(Self::ChaCha8),
            "chacha12" | "chacha12rng" => Some(Self::ChaCha12),
            "chacha20" | "chacha20rng" | "chacha" => Some(Self::ChaCha20),
            _ => None,
        }
    }
}

/// A concrete seed and RNG method accepted by every PiP stochastic API.
///
/// This type contains no defaults or unresolved options. In a workflow-driven
/// application, adapt the workflow seed into this type at the application
/// boundary and retain the value as part of the run provenance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedRng {
    seed: u64,
    method: RngMethod,
}

impl ResolvedRng {
    /// Constructs reproducible randomness from an explicit seed and method.
    pub const fn new(seed: u64, method: RngMethod) -> Self {
        Self { seed, method }
    }

    /// Explicitly chooses a seed from host entropy.
    pub fn from_entropy(method: RngMethod) -> Self {
        Self::new(rand::random(), method)
    }

    pub const fn seed(self) -> u64 {
        self.seed
    }

    pub const fn method(self) -> RngMethod {
        self.method
    }

    /// Returns the seed in the stable encoding reported by the method.
    pub fn encode_seed(self) -> String {
        self.seed.to_string()
    }

    /// Validates that a component implements this resolved method.
    pub fn ensure_supported(
        self,
        component: &'static str,
        supported_methods: &[RngMethod],
    ) -> Result<Self, RngError> {
        if supported_methods.contains(&self.method) {
            Ok(self)
        } else {
            Err(RngError::UnsupportedMethod {
                component,
                method: self.method,
            })
        }
    }
}

/// Invalid use of a resolved RNG method.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RngError {
    UnsupportedMethod {
        component: &'static str,
        method: RngMethod,
    },
}

impl fmt::Display for RngError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedMethod { component, method } => write!(
                formatter,
                "RNG method `{}` is not supported by {component}",
                method.name()
            ),
        }
    }
}

impl std::error::Error for RngError {}

/// Schedule-independent indexed randomness.
///
/// Each value is a pure function of the resolved RNG and the five supplied
/// coordinates, so results do not depend on Rayon scheduling.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct IndexedRng(ResolvedRng);

impl IndexedRng {
    pub fn new(rng: ResolvedRng) -> Result<Self, RngError> {
        rng.ensure_supported("IndexedRng", &[RngMethod::IndexedSplitMix64])
            .map(Self)
    }

    pub const fn resolved_rng(self) -> ResolvedRng {
        self.0
    }

    /// Maps one random coordinate to a uniform floating-point value in `[0, 1)`.
    pub fn unit_f64(self, step: u64, domain: u64, item: u64, component: u64, draw: u64) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.indexed_word(step, domain, item, component, draw) >> 11) as f64) * SCALE
    }

    /// Maps indexed words uniformly into `0..upper` using Lemire rejection.
    pub fn uniform_index(
        self,
        step: u64,
        domain: u64,
        item: u64,
        component: u64,
        upper: usize,
    ) -> Option<usize> {
        if upper == 0 {
            return None;
        }
        let bound = upper as u64;
        let threshold = bound.wrapping_neg() % bound;
        let mut draw = 0;
        loop {
            let word = self.indexed_word(step, domain, item, component, draw);
            let product = u128::from(word) * u128::from(bound);
            if (product as u64) >= threshold {
                return Some((product >> 64) as usize);
            }
            draw = draw.wrapping_add(1);
        }
    }

    /// Returns one deterministic standard-normal sample via Box-Muller.
    pub fn standard_normal(self, step: u64, domain: u64, item: u64, component: u64) -> f64 {
        const DENOMINATOR: f64 = (1_u64 << 53) as f64;
        let word = self.indexed_word(step, domain, item, component, 0) >> 11;
        let u1 = ((word as f64) + 0.5) / DENOMINATOR;
        let u2 = self.unit_f64(step, domain, item, component, 1);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    pub(crate) fn indexed_word(
        self,
        step: u64,
        domain: u64,
        item: u64,
        component: u64,
        draw: u64,
    ) -> u64 {
        let mut state = splitmix64(self.0.seed ^ 0x6a09_e667_f3bc_c909);
        for value in [step, domain, item, component, draw] {
            state = splitmix64(state ^ splitmix64(value.wrapping_add(0x9e37_79b9_7f4a_7c15)));
        }
        state
    }
}

#[inline]
const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_preserves_resolved_randomness() {
        let rng = ResolvedRng::new(42, RngMethod::ChaCha12);
        let json = serde_json::to_string(&rng).unwrap();
        assert_eq!(serde_json::from_str::<ResolvedRng>(&json).unwrap(), rng);
    }

    #[test]
    fn indexed_rng_rejects_stateful_methods() {
        let error = IndexedRng::new(ResolvedRng::new(3, RngMethod::SmallRng)).unwrap_err();
        assert!(matches!(error, RngError::UnsupportedMethod { .. }));
    }
}

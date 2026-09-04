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
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct IndexedRng(ResolvedRng);

impl<'de> Deserialize<'de> for IndexedRng {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(ResolvedRng::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl IndexedRng {
    /// Validates that the resolved method implements indexed randomness.
    /// Deserialization applies the same validation and retains the wire format.
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
        self.prepare(step, domain).word(item, component, draw)
    }

    /// Hoists the shared seed, step and domain without changing coordinate mixing.
    pub(crate) fn prepare(self, step: u64, domain: u64) -> PreparedIndexedRng {
        let state = splitmix64(self.0.seed ^ 0x6a09_e667_f3bc_c909);
        PreparedIndexedRng(mix_coordinate(mix_coordinate(state, step), domain))
    }
}

/// Operation-local prefix; never serialized or used as a different RNG method.
#[derive(Clone, Copy)]
pub(crate) struct PreparedIndexedRng(u64);

impl PreparedIndexedRng {
    fn word(self, item: u64, component: u64, draw: u64) -> u64 {
        mix_coordinate(
            mix_coordinate(mix_coordinate(self.0, item), component),
            draw,
        )
    }

    /// Fills consecutive flattened coordinates, reusing each item's prefix.
    /// Each result exactly matches IndexedRng::unit_f64 at draw zero. Chunk
    /// boundaries may split an item; no padding or fixed component count is needed.
    pub(crate) fn fill_units(self, values: &mut [f64], start: usize, components: usize) {
        assert!(components > 0);
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        let mut item = start / components;
        let mut component = start % components;
        let mut rest = values;
        while !rest.is_empty() {
            let count = rest.len().min(components - component);
            let (row, tail) = rest.split_at_mut(count);
            let prefix = mix_coordinate(self.0, item as u64);
            for (offset, value) in row.iter_mut().enumerate() {
                let word = mix_coordinate(mix_coordinate(prefix, (component + offset) as u64), 0);
                *value = ((word >> 11) as f64) * SCALE;
            }
            rest = tail;
            item += 1;
            component = 0;
        }
    }
}

#[inline]
fn mix_coordinate(state: u64, coordinate: u64) -> u64 {
    splitmix64(state ^ splitmix64(coordinate.wrapping_add(0x9e37_79b9_7f4a_7c15)))
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
    #[test]
    fn indexed_deserialization_validates_every_method() {
        for method in [
            RngMethod::IndexedSplitMix64,
            RngMethod::Pcg64,
            RngMethod::Pcg64Mcg,
            RngMethod::SmallRng,
            RngMethod::ChaCha8,
            RngMethod::ChaCha12,
            RngMethod::ChaCha20,
        ] {
            let resolved = ResolvedRng::new(42, method);
            let restored =
                serde_json::from_str::<IndexedRng>(&serde_json::to_string(&resolved).unwrap());
            assert_eq!(restored.is_ok(), IndexedRng::new(resolved).is_ok());
            if let Ok(restored) = restored {
                assert_eq!(restored.resolved_rng(), resolved);
                assert_eq!(
                    restored.unit_f64(1, 2, 3, 4, 5),
                    IndexedRng::new(resolved).unwrap().unit_f64(1, 2, 3, 4, 5)
                );
            }
        }
    }

    #[test]
    fn batched_words_preserve_original_coordinate_mapping() {
        fn original(seed: u64, coordinates: [u64; 5]) -> u64 {
            let mut state = splitmix64(seed ^ 0x6a09_e667_f3bc_c909);
            for value in coordinates {
                state = splitmix64(state ^ splitmix64(value.wrapping_add(0x9e37_79b9_7f4a_7c15)));
            }
            state
        }
        for seed in [0, 42, u64::MAX] {
            let rng =
                IndexedRng::new(ResolvedRng::new(seed, RngMethod::IndexedSplitMix64)).unwrap();
            for components in [1, 2, 3, 8, 17] {
                for start in [0, 1, 13, 129] {
                    let mut values = [0.0; 131];
                    rng.prepare(u64::MAX, 7)
                        .fill_units(&mut values, start, components);
                    for (offset, value) in values.into_iter().enumerate() {
                        let index = start + offset;
                        let expected = original(
                            seed,
                            [
                                u64::MAX,
                                7,
                                (index / components) as u64,
                                (index % components) as u64,
                                0,
                            ],
                        );
                        assert_eq!(value, ((expected >> 11) as f64) / ((1_u64 << 53) as f64));
                    }
                }
            }
        }
    }
}

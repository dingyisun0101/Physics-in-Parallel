//! Unified configuration and provenance for PiP-owned randomness.
//!
//! Every public stochastic PiP API accepts [`RngConfig`] as its only random
//! configuration input. Sampling implementations remain specialized: indexed
//! randomness has no cursor, while tensor fillers own stateful generators.

use std::fmt;
use std::num::NonZeroUsize;

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
            Self::SmallRng => "rand-0.10.1",
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

/// Complete optional RNG configuration accepted by every public stochastic API.
///
/// Missing values select the receiving component's documented defaults. Once
/// a component resolves this configuration, it retains the generated seed and
/// selected method so callers can record exact provenance.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RngConfig {
    seed: Option<u64>,
    method: Option<RngMethod>,
    parallel_streams: Option<NonZeroUsize>,
}

impl RngConfig {
    /// Creates the sole PiP randomness configuration object.
    pub const fn new(
        seed: Option<u64>,
        method: Option<RngMethod>,
        parallel_streams: Option<NonZeroUsize>,
    ) -> Self {
        Self {
            seed,
            method,
            parallel_streams,
        }
    }

    pub const fn seed(self) -> Option<u64> {
        self.seed
    }

    pub const fn method(self) -> Option<RngMethod> {
        self.method
    }

    pub const fn parallel_streams(self) -> Option<NonZeroUsize> {
        self.parallel_streams
    }

    /// Returns the resolved seed as stable decimal provenance.
    pub fn encode_seed(self) -> Option<String> {
        self.seed.map(|seed| seed.to_string())
    }

    /// Resolves defaults and validates this configuration for one component.
    ///
    /// This is primarily useful to downstream plugin authors. Ordinary users
    /// pass `RngConfig` directly to a stochastic PiP API and read the resolved
    /// configuration back from that component.
    pub fn resolve_for(
        self,
        component: &'static str,
        default_method: RngMethod,
        supported_methods: &[RngMethod],
        default_parallel_streams: Option<NonZeroUsize>,
    ) -> Result<Self, RngConfigError> {
        let method = self.method.unwrap_or(default_method);
        if !supported_methods.contains(&method) {
            return Err(RngConfigError::UnsupportedMethod { component, method });
        }
        if self.parallel_streams.is_some() && default_parallel_streams.is_none() {
            return Err(RngConfigError::ParallelStreamsUnsupported { component });
        }
        Ok(Self {
            seed: Some(self.seed.unwrap_or_else(rand::random)),
            method: Some(method),
            parallel_streams: self.parallel_streams.or(default_parallel_streams),
        })
    }
}

/// Invalid use of unified RNG configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RngConfigError {
    UnsupportedMethod {
        component: &'static str,
        method: RngMethod,
    },
    ParallelStreamsUnsupported {
        component: &'static str,
    },
}

/// Resolved execution object for schedule-independent indexed randomness.
///
/// Each value is a pure function of the resolved [`RngConfig`] and the five
/// caller-supplied coordinates. Downstream scientific plugins can therefore
/// define stable random domains without maintaining a cursor or depending on
/// Rayon scheduling.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct IndexedRng(RngConfig);

impl IndexedRng {
    /// Resolves unified configuration for indexed SplitMix64 randomness.
    pub fn new(rng: RngConfig) -> Result<Self, RngConfigError> {
        rng.resolve_for(
            "IndexedRng",
            RngMethod::IndexedSplitMix64,
            &[RngMethod::IndexedSplitMix64],
            None,
        )
        .map(Self)
    }

    /// Builds an indexed key from a configuration already resolved by a
    /// lower-level stochastic component.
    pub(crate) fn from_resolved(rng: RngConfig) -> Self {
        debug_assert_eq!(rng.method(), Some(RngMethod::IndexedSplitMix64));
        debug_assert!(rng.seed().is_some());
        Self(rng)
    }

    /// Returns fully resolved configuration for provenance.
    pub const fn rng_config(self) -> RngConfig {
        self.0
    }

    /// Maps one random coordinate to a uniform floating-point value in `[0, 1)`.
    pub fn unit_f64(self, step: u64, domain: u64, item: u64, component: u64, draw: u64) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.indexed_word(step, domain, item, component, draw) >> 11) as f64) * SCALE
    }

    /// Maps indexed words uniformly into `0..upper` using Lemire rejection.
    ///
    /// Returns `None` when `upper` is zero.
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
        let mut state = splitmix64(
            self.rng_config().seed().expect("resolved indexed seed") ^ 0x6a09_e667_f3bc_c909,
        );
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

impl fmt::Display for RngConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedMethod { component, method } => write!(
                formatter,
                "RNG method `{}` is not supported by {component}",
                method.name()
            ),
            Self::ParallelStreamsUnsupported { component } => write!(
                formatter,
                "parallel RNG stream configuration is not supported by {component}"
            ),
        }
    }
}

impl std::error::Error for RngConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_preserves_one_unified_configuration() {
        let config = RngConfig::new(Some(42), Some(RngMethod::ChaCha12), NonZeroUsize::new(4));
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(serde_json::from_str::<RngConfig>(&json).unwrap(), config);
    }

    #[test]
    fn resolution_fills_defaults_and_rejects_unsupported_options() {
        let resolved = RngConfig::default()
            .resolve_for(
                "test",
                RngMethod::SmallRng,
                &[RngMethod::SmallRng],
                NonZeroUsize::new(8),
            )
            .unwrap();
        assert!(resolved.seed().is_some());
        assert_eq!(resolved.method(), Some(RngMethod::SmallRng));
        assert_eq!(resolved.parallel_streams(), NonZeroUsize::new(8));

        let error = RngConfig::new(None, Some(RngMethod::IndexedSplitMix64), None)
            .resolve_for(
                "test",
                RngMethod::SmallRng,
                &[RngMethod::SmallRng],
                NonZeroUsize::new(8),
            )
            .unwrap_err();
        assert!(matches!(error, RngConfigError::UnsupportedMethod { .. }));
    }
}

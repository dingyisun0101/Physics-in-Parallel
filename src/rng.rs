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

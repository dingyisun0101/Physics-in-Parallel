/*!
Power-law pair-interaction parameter payload.

Purpose:
`PowerLawDecay` stores the parameters for a generic pairwise power law. It is
only data plus validation. Application code decides whether `k`, `alpha`, and
`range` represent a force, rate, potential, or another model-specific
quantity.
*/

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

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

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PowerLawSerde {
    k: f64,
    alpha: f64,
    range: Option<PowerLawRange>,
}

impl Serialize for PowerLawDecay {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate().map_err(serde::ser::Error::custom)?;
        PowerLawSerde {
            k: self.k,
            alpha: self.alpha,
            range: self.range,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PowerLawDecay {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = PowerLawSerde::deserialize(deserializer)?;
        Self::new(value.k, value.alpha, value.range).map_err(serde::de::Error::custom)
    }
}

impl PowerLawDecay {
    /// Builds a validated power-law payload.
    pub fn new(k: f64, alpha: f64, range: Option<PowerLawRange>) -> Result<Self, PowerLawError> {
        let payload = Self { k, alpha, range };
        payload.validate()?;
        Ok(payload)
    }

    /// Validates power-law parameters.
    pub fn validate(&self) -> Result<(), PowerLawError> {
        if !self.k.is_finite() {
            return Err(PowerLawError::InvalidStrength { k: self.k });
        }
        if !self.alpha.is_finite() {
            return Err(PowerLawError::InvalidExponent { alpha: self.alpha });
        }
        if let Some((min, max)) = self.range
            && (!min.is_finite() || !max.is_finite() || min < 0.0 || max < min)
        {
            return Err(PowerLawError::InvalidRange { min, max });
        }
        Ok(())
    }
}

/// Errors returned by power-law validation.
#[derive(Debug, Clone, PartialEq)]
pub enum PowerLawError {
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

impl fmt::Display for PowerLawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStrength { k } => {
                write!(f, "power-law strength must be finite; got {k}")
            }
            Self::InvalidExponent { alpha } => {
                write!(f, "power-law exponent must be finite; got {alpha}")
            }
            Self::InvalidRange { min, max } => write!(
                f,
                "power-law active range must satisfy finite 0 <= min <= max; got min={min}, max={max}"
            ),
        }
    }
}

impl std::error::Error for PowerLawError {}

/*!
Hooke-law spring parameter payload.

Purpose:
`Spring` stores the parameters for a pairwise Hooke-law interaction. It is only
data plus validation. It does not know whether the endpoints are particles,
lattice sites, graph nodes, or any future model object.
*/

use core::fmt;

/// Optional distance window `(min, max)` where this spring is active.
pub type SpringCutoff = (f64, f64);

/// Per-edge spring payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spring {
    /// Spring constant.
    pub k: f64,
    /// Unstretched length.
    pub l_0: f64,
    /// Optional pair-distance cutoff `(min, max)`.
    pub cutoff: Option<SpringCutoff>,
}

impl Spring {
    /// Builds a validated spring payload.
    pub fn new(k: f64, l_0: f64, cutoff: Option<SpringCutoff>) -> Result<Self, SpringLawError> {
        let spring = Self { k, l_0, cutoff };
        spring.validate()?;
        Ok(spring)
    }

    /// Validates physical spring parameters.
    pub fn validate(&self) -> Result<(), SpringLawError> {
        if !self.k.is_finite() {
            return Err(SpringLawError::InvalidSpringConstant { k: self.k });
        }
        if !self.l_0.is_finite() || self.l_0 < 0.0 {
            return Err(SpringLawError::InvalidRestLength { l_0: self.l_0 });
        }
        if let Some((min, max)) = self.cutoff
            && (!min.is_finite() || !max.is_finite() || min < 0.0 || max < min)
        {
            return Err(SpringLawError::InvalidCutoff { min, max });
        }
        Ok(())
    }
}

/// Errors returned by spring-law validation.
#[derive(Debug, Clone, PartialEq)]
pub enum SpringLawError {
    /// Spring constant is not finite.
    InvalidSpringConstant {
        /// Invalid spring constant.
        k: f64,
    },
    /// Rest length is not finite or is negative.
    InvalidRestLength {
        /// Invalid rest length.
        l_0: f64,
    },
    /// Cutoff bounds are not finite, negative, or ordered incorrectly.
    InvalidCutoff {
        /// Lower active distance.
        min: f64,
        /// Upper active distance.
        max: f64,
    },
}

impl fmt::Display for SpringLawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpringConstant { k } => {
                write!(f, "spring constant must be finite; got {k}")
            }
            Self::InvalidRestLength { l_0 } => {
                write!(
                    f,
                    "spring rest length must be finite and non-negative; got {l_0}"
                )
            }
            Self::InvalidCutoff { min, max } => write!(
                f,
                "spring cutoff must satisfy finite 0 <= min <= max; got min={min}, max={max}"
            ),
        }
    }
}

impl std::error::Error for SpringLawError {}

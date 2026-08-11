/*!
Indexed randomness for scientific algorithms.

Every generated value is a pure function of an explicit key, sweep, random
domain, pair index, component, and draw index. Work can therefore be divided
across Rayon workers in any way without changing the generated pair batch.

The generator is intended for reproducible simulation, not cryptography.
Changing its mapping is a scientific format change and requires incrementing
[`INDEXED_RANDOM_VERSION`].
*/

use serde::{Deserialize, Serialize};

/// Stable method identifier suitable for scientific provenance records.
pub const INDEXED_RANDOM_METHOD: &str = "splitmix64_indexed";

/// Version of the indexed tuple-to-random-word mapping.
pub const INDEXED_RANDOM_VERSION: &str = "1";

/// Stable encoding used when persisting [`RandomKey`].
pub const INDEXED_RANDOM_KEY_ENCODING: &str = "u64_decimal";

/// Root key for indexed scientific randomness.
///
/// One interface covers deterministic and entropy-backed construction. The
/// resolved key is always retained and can be persisted for reproducibility.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct RandomKey(u64);

impl RandomKey {
    /// Creates a key from an optional seed.
    ///
    /// `Some(seed)` is exactly reproducible. `None` draws one seed from host
    /// entropy; callers can recover and record it through [`Self::value`].
    pub fn new(seed: Option<u64>) -> Self {
        Self(seed.unwrap_or_else(rand::random))
    }

    /// Returns the numeric key without changing its stable interpretation.
    pub const fn value(self) -> u64 {
        self.0
    }

    /// Encodes the key using [`INDEXED_RANDOM_KEY_ENCODING`].
    pub fn encode(self) -> String {
        self.0.to_string()
    }

    /// Produces one stable word from the complete random coordinate.
    pub fn word(self, step: u64, domain: u64, item: u64, component: u64, draw: u64) -> u64 {
        indexed_word(self, step, domain, item, component, draw)
    }

    /// Maps one random coordinate to a uniform floating-point value in `[0, 1)`.
    pub fn unit_f64(self, step: u64, domain: u64, item: u64, component: u64, draw: u64) -> f64 {
        unit_f64(self, step, domain, item, component, draw)
    }

    /// Maps one random coordinate uniformly into `0..upper`.
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
        (upper > 0).then(|| uniform_index(self, step, domain, item, component, upper))
    }

    /// Returns one deterministic standard-normal sample via Box-Muller.
    pub fn standard_normal(self, step: u64, domain: u64, item: u64, component: u64) -> f64 {
        standard_normal(self, step, domain, item, component)
    }
}

pub(crate) const DOMAIN_SOURCE_COORDINATE: u64 = 0x20dd_7f45_5d92_4a31;
pub(crate) const DOMAIN_HAAR_COMPONENT: u64 = 0x64a4_a4fe_1a89_827d;
pub(crate) const DOMAIN_KERNEL_SAMPLE: u64 = 0xbeb3_9487_4b9c_a7f5;

/// Produces one stable word from the complete random coordinate.
pub(crate) fn indexed_word(
    key: RandomKey,
    sweep: u64,
    domain: u64,
    item: u64,
    component: u64,
    draw: u64,
) -> u64 {
    let mut state = splitmix64(key.value() ^ 0x6a09_e667_f3bc_c909);
    for value in [sweep, domain, item, component, draw] {
        state = splitmix64(state ^ splitmix64(value.wrapping_add(0x9e37_79b9_7f4a_7c15)));
    }
    state
}

/// Maps one indexed word to a uniform floating-point value in `[0, 1)`.
pub(crate) fn unit_f64(
    key: RandomKey,
    sweep: u64,
    domain: u64,
    item: u64,
    component: u64,
    draw: u64,
) -> f64 {
    const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
    ((indexed_word(key, sweep, domain, item, component, draw) >> 11) as f64) * SCALE
}

/// Maps indexed words uniformly into `0..upper` using Lemire rejection.
pub(crate) fn uniform_index(
    key: RandomKey,
    sweep: u64,
    domain: u64,
    item: u64,
    component: u64,
    upper: usize,
) -> usize {
    debug_assert!(upper > 0);
    let bound = upper as u64;
    let threshold = bound.wrapping_neg() % bound;
    let mut draw = 0;
    loop {
        let word = indexed_word(key, sweep, domain, item, component, draw);
        let product = u128::from(word) * u128::from(bound);
        if (product as u64) >= threshold {
            return (product >> 64) as usize;
        }
        draw = draw.wrapping_add(1);
    }
}

/// Returns one deterministic standard-normal sample via Box-Muller.
pub(crate) fn standard_normal(
    key: RandomKey,
    sweep: u64,
    domain: u64,
    item: u64,
    component: u64,
) -> f64 {
    const DENOMINATOR: f64 = (1_u64 << 53) as f64;
    let word = indexed_word(key, sweep, domain, item, component, 0) >> 11;
    let u1 = ((word as f64) + 0.5) / DENOMINATOR;
    let u2 = unit_f64(key, sweep, domain, item, component, 1);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
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
    fn complete_random_coordinate_changes_the_word() {
        let key = RandomKey::new(Some(17));
        let base = indexed_word(key, 2, 3, 4, 5, 6);
        assert_eq!(base, 0x995b_eef1_54ed_1885);
        assert_ne!(base, indexed_word(RandomKey::new(Some(18)), 2, 3, 4, 5, 6));
        assert_ne!(base, indexed_word(key, 3, 3, 4, 5, 6));
        assert_ne!(base, indexed_word(key, 2, 4, 4, 5, 6));
        assert_ne!(base, indexed_word(key, 2, 3, 5, 5, 6));
        assert_ne!(base, indexed_word(key, 2, 3, 4, 6, 6));
        assert_ne!(base, indexed_word(key, 2, 3, 4, 5, 7));
    }

    #[test]
    fn public_identity_is_stable() {
        let key = RandomKey::new(Some(12_345));
        assert_eq!(key.encode(), "12345");
        assert_eq!(INDEXED_RANDOM_METHOD, "splitmix64_indexed");
        assert_eq!(INDEXED_RANDOM_VERSION, "1");
        assert_eq!(INDEXED_RANDOM_KEY_ENCODING, "u64_decimal");
    }
}

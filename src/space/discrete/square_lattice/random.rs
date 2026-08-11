/*!
Indexed randomness for scientific algorithms.

Every generated value is a pure function of an explicit key, sweep, random
domain, pair index, component, and draw index. Work can therefore be divided
across Rayon workers in any way without changing the generated pair batch.

The generator is intended for reproducible simulation, not cryptography.
Changing its mapping is a scientific format change and requires updating
[`RngMethod::version`].
*/

use serde::{Deserialize, Serialize};

use crate::rng::{RngConfig, RngConfigError, RngMethod};

/// Resolved execution object for indexed scientific randomness.
///
/// One interface covers deterministic and entropy-backed construction. The
/// resolved key is always retained and can be persisted for reproducibility.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub(crate) struct IndexedRng(RngConfig);

impl IndexedRng {
    /// Resolves unified configuration for indexed SplitMix64 randomness.
    pub(crate) fn new(rng: RngConfig) -> Result<Self, RngConfigError> {
        rng.resolve_for(
            "IndexedRng",
            RngMethod::IndexedSplitMix64,
            &[RngMethod::IndexedSplitMix64],
            None,
        )
        .map(Self)
    }

    /// Returns fully resolved configuration for provenance.
    pub(crate) const fn rng_config(self) -> RngConfig {
        self.0
    }

    /// Maps one random coordinate to a uniform floating-point value in `[0, 1)`.
    pub(crate) fn unit_f64(
        self,
        step: u64,
        domain: u64,
        item: u64,
        component: u64,
        draw: u64,
    ) -> f64 {
        unit_f64(self, step, domain, item, component, draw)
    }

    /// Maps one random coordinate uniformly into `0..upper`.
    ///
    /// Returns `None` when `upper` is zero.
    pub(crate) fn uniform_index(
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
    pub(crate) fn standard_normal(self, step: u64, domain: u64, item: u64, component: u64) -> f64 {
        standard_normal(self, step, domain, item, component)
    }
}

pub(crate) const DOMAIN_SOURCE_COORDINATE: u64 = 0x20dd_7f45_5d92_4a31;
pub(crate) const DOMAIN_HAAR_COMPONENT: u64 = 0x64a4_a4fe_1a89_827d;
pub(crate) const DOMAIN_KERNEL_SAMPLE: u64 = 0xbeb3_9487_4b9c_a7f5;

/// Produces one stable word from the complete random coordinate.
pub(crate) fn indexed_word(
    key: IndexedRng,
    sweep: u64,
    domain: u64,
    item: u64,
    component: u64,
    draw: u64,
) -> u64 {
    let mut state =
        splitmix64(key.rng_config().seed().expect("resolved indexed seed") ^ 0x6a09_e667_f3bc_c909);
    for value in [sweep, domain, item, component, draw] {
        state = splitmix64(state ^ splitmix64(value.wrapping_add(0x9e37_79b9_7f4a_7c15)));
    }
    state
}

/// Maps one indexed word to a uniform floating-point value in `[0, 1)`.
pub(crate) fn unit_f64(
    key: IndexedRng,
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
    key: IndexedRng,
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
    key: IndexedRng,
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
        let key = IndexedRng::new(RngConfig::new(Some(17), None, None)).unwrap();
        let base = indexed_word(key, 2, 3, 4, 5, 6);
        assert_eq!(base, 0x995b_eef1_54ed_1885);
        assert_ne!(
            base,
            indexed_word(
                IndexedRng::new(RngConfig::new(Some(18), None, None)).unwrap(),
                2,
                3,
                4,
                5,
                6
            )
        );
        assert_ne!(base, indexed_word(key, 3, 3, 4, 5, 6));
        assert_ne!(base, indexed_word(key, 2, 4, 4, 5, 6));
        assert_ne!(base, indexed_word(key, 2, 3, 5, 5, 6));
        assert_ne!(base, indexed_word(key, 2, 3, 4, 6, 6));
        assert_ne!(base, indexed_word(key, 2, 3, 4, 5, 7));
    }

    #[test]
    fn public_identity_is_stable() {
        let key = IndexedRng::new(RngConfig::new(Some(12_345), None, None)).unwrap();
        let config = key.rng_config();
        assert_eq!(config.encode_seed().as_deref(), Some("12345"));
        assert_eq!(config.method().unwrap().name(), "splitmix64_indexed");
        assert_eq!(config.method().unwrap().version(), "1");
        assert_eq!(config.method().unwrap().seed_encoding(), "u64_decimal");
    }
}

/*!
Indexed randomness for scientific algorithms.

Every generated value is a pure function of an explicit key, sweep, random
domain, pair index, component, and draw index. Work can therefore be divided
across Rayon workers in any way without changing the generated pair batch.

The generator is intended for reproducible simulation, not cryptography.
Changing its mapping is a scientific format change and requires updating
[`RngMethod::version`].
*/

pub(crate) use crate::rng::IndexedRng;

pub(crate) const DOMAIN_SOURCE_COORDINATE: u64 = 0x20dd_7f45_5d92_4a31;
pub(crate) const DOMAIN_INDEPENDENT_SOURCE_SITE: u64 = 0x5fc7_f80b_2257_ae21;
pub(crate) const DOMAIN_INDEPENDENT_TARGET_SITE: u64 = 0xe895_319d_634c_c4d3;
pub(crate) const DOMAIN_HAAR_COMPONENT: u64 = 0x64a4_a4fe_1a89_827d;
pub(crate) const DOMAIN_KERNEL_SAMPLE: u64 = 0xbeb3_9487_4b9c_a7f5;

/// Produces one stable word from the complete random coordinate.
#[cfg(test)]
pub(crate) fn indexed_word(
    key: IndexedRng,
    sweep: u64,
    domain: u64,
    item: u64,
    component: u64,
    draw: u64,
) -> u64 {
    key.indexed_word(sweep, domain, item, component, draw)
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
    key.unit_f64(sweep, domain, item, component, draw)
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
    key.uniform_index(sweep, domain, item, component, upper)
        .expect("positive upper bound")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::RngConfig;

    #[test]
    fn complete_random_coordinate_changes_the_word() {
        let key = IndexedRng::new(RngConfig::new(Some(17), None)).unwrap();
        let base = indexed_word(key, 2, 3, 4, 5, 6);
        assert_eq!(base, 0x995b_eef1_54ed_1885);
        assert_ne!(
            base,
            indexed_word(
                IndexedRng::new(RngConfig::new(Some(18), None)).unwrap(),
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
        let key = IndexedRng::new(RngConfig::new(Some(12_345), None)).unwrap();
        let config = key.rng_config();
        assert_eq!(config.encode_seed().as_deref(), Some("12345"));
        assert_eq!(config.method().unwrap().name(), "splitmix64_indexed");
        assert_eq!(config.method().unwrap().version(), "1");
        assert_eq!(config.method().unwrap().seed_encoding(), "u64_decimal");
    }
}

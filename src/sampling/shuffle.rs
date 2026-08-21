//! Indexed in-place permutation used by PiP-owned scientific containers.

use crate::rng::{IndexedRng, RngConfig, RngConfigError};

const DOMAIN_INDEXED_SHUFFLE: u64 = 0x899f_2c5e_12bd_ba4d;

/// Applies an unbiased Fisher-Yates permutation and returns resolved RNG provenance.
pub(crate) fn shuffle_slice_indexed<T>(
    values: &mut [T],
    rng: RngConfig,
) -> Result<RngConfig, RngConfigError> {
    let key = IndexedRng::new(rng)?;
    for (draw, upper) in (2..=values.len()).rev().enumerate() {
        let selected = key
            .uniform_index(0, DOMAIN_INDEXED_SHUFFLE, draw as u64, 0, upper)
            .expect("Fisher-Yates upper bound is positive");
        values.swap(upper - 1, selected);
    }
    Ok(key.rng_config())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_shuffle_replays_and_preserves_the_multiset() {
        let source: Vec<usize> = (0..128).map(|index| index % 7).collect();
        let mut first = source.clone();
        let mut second = source.clone();
        let rng = RngConfig::new(Some(41), None);
        let resolved = shuffle_slice_indexed(&mut first, rng).unwrap();
        assert_eq!(resolved.seed(), Some(41));
        shuffle_slice_indexed(&mut second, rng).unwrap();
        assert_eq!(first, second);
        assert_ne!(first, source);

        let mut expected = source;
        let mut actual = first;
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(actual, expected);
    }
}

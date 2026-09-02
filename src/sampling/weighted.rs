//! Mutable exact weighted selection backed by a Fenwick tree.

use std::error::Error;
use std::fmt;

/// A mutable collection of nonnegative integer weights supporting exact
/// order-statistic selection and logarithmic updates.
///
/// This type deliberately owns no random generator. Callers draw an integer in
/// `0..total()` with their chosen PiP RNG and pass it to [`Self::select`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DynamicWeightedIndex {
    weights: Vec<usize>,
    /// One-indexed Fenwick partial sums; slot zero is unused.
    tree: Vec<usize>,
    total: usize,
}

impl DynamicWeightedIndex {
    /// Builds an index in linear time from one weight per selectable item.
    pub fn new(weights: &[usize]) -> Result<Self, DynamicWeightedIndexError> {
        if weights.is_empty() {
            return Err(DynamicWeightedIndexError::EmptyWeights);
        }
        let total = weights.iter().try_fold(0usize, |sum, &weight| {
            sum.checked_add(weight)
                .ok_or(DynamicWeightedIndexError::WeightOverflow)
        })?;
        let mut tree = vec![0usize; weights.len() + 1];
        for (index, &weight) in weights.iter().enumerate() {
            let slot = index + 1;
            tree[slot] = tree[slot]
                .checked_add(weight)
                .ok_or(DynamicWeightedIndexError::WeightOverflow)?;
            let parent = slot + low_bit(slot);
            if parent < tree.len() {
                tree[parent] = tree[parent]
                    .checked_add(tree[slot])
                    .ok_or(DynamicWeightedIndexError::WeightOverflow)?;
            }
        }
        Ok(Self {
            weights: weights.to_vec(),
            tree,
            total,
        })
    }

    /// Number of selectable items, including items whose current weight is zero.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Returns whether the index contains no items.
    ///
    /// Construction rejects empty input, so this is always false for a valid
    /// instance and is supplied for ordinary collection ergonomics.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Sum of all current weights.
    pub const fn total(&self) -> usize {
        self.total
    }

    /// Returns one item's current weight.
    pub fn weight(&self, index: usize) -> Option<usize> {
        self.weights.get(index).copied()
    }

    /// Replaces one weight while retaining exact checked partial sums.
    pub fn set_weight(
        &mut self,
        index: usize,
        weight: usize,
    ) -> Result<(), DynamicWeightedIndexError> {
        let Some(&previous) = self.weights.get(index) else {
            return Err(DynamicWeightedIndexError::IndexOutOfBounds {
                index,
                len: self.len(),
            });
        };
        if weight == previous {
            return Ok(());
        }

        if weight > previous {
            let increase = weight - previous;
            let total = self
                .total
                .checked_add(increase)
                .ok_or(DynamicWeightedIndexError::WeightOverflow)?;
            let mut slot = index + 1;
            while slot < self.tree.len() {
                self.tree[slot] = self.tree[slot]
                    .checked_add(increase)
                    .ok_or(DynamicWeightedIndexError::WeightOverflow)?;
                slot += low_bit(slot);
            }
            self.total = total;
        } else {
            let decrease = previous - weight;
            let mut slot = index + 1;
            while slot < self.tree.len() {
                self.tree[slot] -= decrease;
                slot += low_bit(slot);
            }
            self.total -= decrease;
        }
        self.weights[index] = weight;
        Ok(())
    }

    /// Selects the item containing the zero-based order in the weighted range.
    ///
    /// Returns `None` when `order >= total()`; zero-weight items are never
    /// selected.
    pub fn select(&self, order: usize) -> Option<usize> {
        if order >= self.total {
            return None;
        }
        Some(self.select_valid_order(order))
    }

    /// Selects by weighted order after removing one item's complete weight.
    ///
    /// `order` is interpreted in `0..total() - weight(excluded)`. Returns
    /// `None` for an invalid excluded index or an out-of-range order.
    pub fn select_excluding(&self, order: usize, excluded: usize) -> Option<usize> {
        let excluded_weight = self.weight(excluded)?;
        let available = self.total - excluded_weight;
        if order >= available {
            return None;
        }
        let before_excluded = self.prefix_sum(excluded);
        let full_order = if order >= before_excluded {
            order + excluded_weight
        } else {
            order
        };
        self.select(full_order)
    }

    fn prefix_sum(&self, end: usize) -> usize {
        let mut slot = end;
        let mut sum = 0usize;
        while slot > 0 {
            sum += self.tree[slot];
            slot -= low_bit(slot);
        }
        sum
    }

    fn select_valid_order(&self, order: usize) -> usize {
        let mut index = 0usize;
        let mut accumulated = 0usize;
        let mut step = highest_power_of_two_at_most(self.len());
        while step > 0 {
            let candidate = index + step;
            if candidate < self.tree.len() && accumulated + self.tree[candidate] <= order {
                index = candidate;
                accumulated += self.tree[candidate];
            }
            step >>= 1;
        }
        index
    }
}

#[inline]
const fn low_bit(value: usize) -> usize {
    value.isolate_lowest_one()
}

fn highest_power_of_two_at_most(value: usize) -> usize {
    1usize << (usize::BITS - 1 - value.leading_zeros())
}

/// Invalid dynamic weighted-index construction or update.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DynamicWeightedIndexError {
    EmptyWeights,
    WeightOverflow,
    IndexOutOfBounds { index: usize, len: usize },
}

impl fmt::Display for DynamicWeightedIndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyWeights => formatter.write_str("dynamic weighted index requires an item"),
            Self::WeightOverflow => formatter.write_str("dynamic weight total exceeds usize"),
            Self::IndexOutOfBounds { index, len } => {
                write!(formatter, "weighted index {index} is outside 0..{len}")
            }
        }
    }
}

impl Error for DynamicWeightedIndexError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_select(weights: &[usize], mut order: usize) -> Option<usize> {
        if order >= weights.iter().sum() {
            return None;
        }
        for (index, &weight) in weights.iter().enumerate() {
            if order < weight {
                return Some(index);
            }
            order -= weight;
        }
        None
    }

    fn assert_matches_linear(index: &DynamicWeightedIndex, weights: &[usize]) {
        assert_eq!(index.total(), weights.iter().sum::<usize>());
        for order in 0..=index.total() {
            assert_eq!(index.select(order), linear_select(weights, order));
        }
    }

    #[test]
    fn selection_matches_linear_cumulative_weights() {
        let weights = [0, 3, 1, 0, 5, 2];
        let index = DynamicWeightedIndex::new(&weights).unwrap();
        assert_matches_linear(&index, &weights);
    }

    #[test]
    fn updates_retain_exact_selection() {
        let mut weights = vec![2, 0, 4, 1];
        let mut index = DynamicWeightedIndex::new(&weights).unwrap();
        for (item, weight) in [(1, 5), (2, 1), (0, 0), (3, 9)] {
            weights[item] = weight;
            index.set_weight(item, weight).unwrap();
            assert_matches_linear(&index, &weights);
        }
    }

    #[test]
    fn excluded_selection_matches_filtered_linear_weights() {
        let weights = [2, 0, 4, 3];
        let index = DynamicWeightedIndex::new(&weights).unwrap();
        for excluded in 0..weights.len() {
            let mut filtered = weights;
            filtered[excluded] = 0;
            let available = filtered.iter().sum();
            for order in 0..=available {
                assert_eq!(
                    index.select_excluding(order, excluded),
                    linear_select(&filtered, order)
                );
            }
        }
        assert_eq!(index.select_excluding(0, weights.len()), None);
    }

    #[test]
    fn zero_total_has_no_selection() {
        let index = DynamicWeightedIndex::new(&[0, 0, 0]).unwrap();
        assert_eq!(index.total(), 0);
        assert_eq!(index.select(0), None);
        assert_eq!(index.select_excluding(0, 1), None);
    }

    #[test]
    fn construction_and_updates_report_domain_errors() {
        assert_eq!(
            DynamicWeightedIndex::new(&[]),
            Err(DynamicWeightedIndexError::EmptyWeights)
        );
        assert_eq!(
            DynamicWeightedIndex::new(&[usize::MAX, 1]),
            Err(DynamicWeightedIndexError::WeightOverflow)
        );
        let mut index = DynamicWeightedIndex::new(&[usize::MAX, 0]).unwrap();
        assert_eq!(
            index.set_weight(1, 1),
            Err(DynamicWeightedIndexError::WeightOverflow)
        );
        assert_eq!(index.weight(2), None);
        assert_eq!(
            index.set_weight(2, 1),
            Err(DynamicWeightedIndexError::IndexOutOfBounds { index: 2, len: 2 })
        );
    }
}

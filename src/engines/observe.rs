/*!
Generic observation reducers.

Purpose:
Reducers combine a batch of observed values into one summary value. They are
model-agnostic: a reducer does not know whether values came from particles,
fields, lattice sites, or any other simulation object.
*/

/// Reduces a batch of observed values into one aggregate value.
pub trait Reducer<T> {
    /// Reduces `values` into one aggregate.
    fn reduce(&self, values: &[T]) -> T;
}

/// Arithmetic mean reducer for scalar observations.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanReducer;

impl Reducer<f64> for MeanReducer {
    fn reduce(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / (values.len() as f64)
    }
}

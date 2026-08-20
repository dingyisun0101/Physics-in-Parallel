use physics_in_parallel::prelude::advanced::{MeanReducer, Reducer};

#[test]
fn mean_reducer_handles_empty_and_nonempty_inputs() {
    let reducer = MeanReducer;

    assert_eq!(reducer.reduce(&[]), 0.0);
    assert_eq!(reducer.reduce(&[1.0, 2.0, 6.0]), 3.0);
}

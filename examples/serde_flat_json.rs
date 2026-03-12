use physics_in_parallel::math::tensor::core::{dense, sparse, tensor_trait::TensorTrait};
use physics_in_parallel::math::tensor::VectorList;
use physics_in_parallel::space::discrete::representation::{Grid, GridConfig, GridInitMethod};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dense_t = dense::Tensor::<f64>::empty(&[2, 2]);
    dense_t.set(&[0, 0], 1.0);
    dense_t.set(&[1, 1], 4.0);

    let mut sparse_t = sparse::Tensor::<f64>::empty(&[2, 3]);
    sparse_t.set(&[0, 1], 2.0);
    sparse_t.set(&[1, 2], 5.0);

    let mut vectors = VectorList::<f64>::empty(3, 2); // dim=3, n=2 -> shape [2, 3]
    vectors.set(0, 0, 1.0);
    vectors.set(0, 1, 2.0);
    vectors.set(0, 2, 3.0);
    vectors.set(1, 0, 4.0);
    vectors.set(1, 1, 5.0);
    vectors.set(1, 2, 6.0);

    let grid = Grid::<usize>::new(
        GridConfig::new(2, 4, true),
        GridInitMethod::Uniform { val: 1 },
    );

    println!(
        "dense tensor JSON:\n{}\n",
        serde_json::to_string_pretty(&dense_t)?
    );
    println!(
        "sparse tensor JSON (serialized densely):\n{}\n",
        serde_json::to_string_pretty(&sparse_t)?
    );
    println!(
        "vector_list JSON:\n{}\n",
        serde_json::to_string_pretty(&vectors)?
    );
    println!("grid JSON:\n{}", serde_json::to_string_pretty(&grid)?);

    Ok(())
}

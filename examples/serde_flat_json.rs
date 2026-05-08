use physics_in_parallel::math::tensor::VectorList;
use physics_in_parallel::math::tensor::{TensorTrait, dense, sparse};
use physics_in_parallel::space::discrete::square_lattice::{
    BoundaryCondition, SquareLattice, SquareLatticeConfig, SquareLatticeInitMethod,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dense_t = dense::Tensor::<f64>::empty(&[2, 2]);
    dense_t.set(&[0, 0], 1.0);
    dense_t.set(&[1, 1], 4.0);

    let mut sparse_t = sparse::Tensor::<f64>::empty(&[2, 3]);
    sparse_t.set(&[0, 1], 2.0);
    sparse_t.set(&[1, 2], 5.0);

    let mut vectors = VectorList::<f64>::empty(3, 2);
    vectors.set_vec(0, &[1.0, 2.0, 3.0]);
    vectors.set_vec(1, &[4.0, 5.0, 6.0]);

    let lattice = SquareLattice::<usize>::new(
        SquareLatticeConfig::new(&vec![4; 2], BoundaryCondition::Periodic),
        SquareLatticeInitMethod::Uniform { val: 1 },
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
    println!("lattice JSON:\n{}", serde_json::to_string_pretty(&lattice)?);

    Ok(())
}

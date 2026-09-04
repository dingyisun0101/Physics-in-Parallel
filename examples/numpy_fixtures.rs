//! Regenerate Python interoperability fixtures with:
//! `cargo run --example numpy_fixtures > python/fixtures.json`.
use physics_in_parallel::prelude::basic::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dense = Tensor::from_values(&[2, 2], Backend::Dense, vec![1_i64, 2, 3, 4])?;
    let sparse = Tensor::from_values(&[2, 2], Backend::Sparse, vec![0.0_f64, 2.0, 0.0, -4.0])?;
    let geometry = SquareLatticeGeometry::new(&[2, 2], BoundaryCondition::Periodic, None)?;
    let lattice = SquareLattice::new(
        geometry,
        SquareLatticeInitMethod::ShuffledValues {
            values: vec![1_i32, 2, 3, 4],
            rng: ResolvedRng::new(17, RngMethod::IndexedSplitMix64),
        },
    )?;
    let documents = vec![
        serde_json::to_value(dense)?,
        serde_json::to_value(sparse)?,
        serde_json::to_value(lattice)?,
    ];
    println!("{}", serde_json::to_string_pretty(&documents)?);
    Ok(())
}

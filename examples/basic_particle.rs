use physics_in_parallel::prelude::basic::{
    ResolvedRng, RngMethod, VectorSamplingMethod, set_max_threads,
};
use physics_in_parallel::prelude::models::{
    ATTR_R, ParticleSelection, Spring, SpringNetwork, create_template, randomize_r,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_max_threads(Some(4))?;

    let mut particles = create_template(2, 2)?;
    let rng = ResolvedRng::new(42, RngMethod::IndexedSplitMix64);
    randomize_r(
        &mut particles,
        VectorSamplingMethod::Uniform {
            low: -1.0,
            high: 1.0,
        },
        rng,
    )?;

    let mut springs = SpringNetwork::new();
    springs.insert((0, 1), Spring::new(1.0, 0.5, None)?)?;
    springs.apply(&mut particles, ParticleSelection::AliveOnly)?;

    println!("positions: {:?}", particles.attribute::<f64>(ATTR_R)?);
    Ok(())
}

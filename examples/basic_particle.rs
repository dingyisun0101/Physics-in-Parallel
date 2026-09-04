//! A complete particle loop with an application-owned Rayon pool.
//! Semi-implicit Euler -> reflecting walls -> Langevin thermostat is the
//! explicitly chosen splitting here; it is not a higher-order integrator.
use physics_in_parallel::prelude::basic::{ReflectBox, ResolvedRng, RngMethod, set_max_threads};
use physics_in_parallel::prelude::models::*;

fn simulate() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut particles = create_template(2, 2)?;
    particles.set_attribute_vector(ATTR_R, 0, &[0.25, 0.5])?;
    particles.set_attribute_vector(ATTR_R, 1, &[0.75, 0.5])?;
    set_mass(&mut particles, 0, 2.0)?;
    let mut springs = SpringNetwork::new();
    springs.insert((0, 1), Spring::new(1.0, 0.4, None)?)?;
    let boundary = ReflectBox::new(&[0.0, 0.0], &[1.0, 1.0])?;
    let mut thermostat = LangevinThermostat::new(
        0.01,
        0.1,
        ResolvedRng::new(42, RngMethod::IndexedSplitMix64),
        ParticleSelection::AliveOnly,
    )?;
    let mut integrator = SemiImplicitEuler;
    for step in 0..20 {
        // Force modules accumulate acceleration; each step starts from zero.
        particles.attribute_mut::<f64>(ATTR_A)?.fill(0.0);
        springs.apply(&mut particles, ParticleSelection::AliveOnly)?;
        integrator.apply(&mut particles, 0.01)?;
        boundary.apply_to_particles(&mut particles)?;
        thermostat.apply(&mut particles, 0.01)?;
        if (step + 1) % 10 == 0 {
            let summary = kinetic_summary(&particles, ParticleSelection::AliveOnly)?;
            println!(
                "step {}: KE={}, temperature={}",
                step + 1,
                summary.energy,
                summary.temperature
            );
        }
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    set_max_threads(Some(4))?;
    pool.install(simulate)
}

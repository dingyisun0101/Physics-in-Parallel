use physics_in_parallel::prelude::basic::{PeriodicBox, RngConfig, VectorSamplingMethod};
use physics_in_parallel::prelude::models::*;

struct DownstreamParticleConfig {
    dimension: usize,
    count: usize,
    box_size: Vec<f64>,
}

impl DownstreamParticleConfig {
    fn position_sampling(&self) -> VectorSamplingMethod<'_> {
        VectorSamplingMethod::UniformCentered {
            box_size: &self.box_size,
        }
    }
}

#[test]
fn downstream_owned_values_construct_and_inspect_the_model_surface() {
    let config = DownstreamParticleConfig {
        dimension: 2,
        count: 3,
        box_size: vec![4.0, 6.0],
    };
    let mut particles = create_template(config.dimension, config.count).unwrap();
    let resolved_rng = randomize_r(
        &mut particles,
        config.position_sampling(),
        RngConfig::new(Some(17), None),
    )
    .unwrap();
    assert_eq!(particles.num_objects(), Some(3));
    assert_eq!(resolved_rng.seed(), Some(17));

    let boundary = PeriodicBox::new(&[-2.0, -3.0], &[2.0, 3.0]).unwrap();
    assert_eq!(boundary.min(), [-2.0, -3.0]);
    assert_eq!(boundary.max(), [2.0, 3.0]);
    boundary.apply_to_particles(&mut particles).unwrap();

    let springs = SpringNetwork::from_records(
        3,
        [SpringRecord {
            i: 0,
            j: 2,
            spring: Spring::new(4.0, 1.5, None).unwrap(),
        }],
    )
    .unwrap();
    let (i, j, spring) = springs.iter_springs().next().unwrap().unwrap();
    assert_eq!((i, j), (0, 2));
    assert_eq!(spring.k, 4.0);

    let power_laws = PowerLawNetwork::from_records(
        3,
        [PowerLawRecord {
            i: 1,
            j: 2,
            law: PowerLawDecay::new(2.0, -2.0, None).unwrap(),
        }],
    )
    .unwrap();
    let (i, j, law) = power_laws.iter_power_laws().next().unwrap().unwrap();
    assert_eq!((i, j), (1, 2));
    assert_eq!(law.alpha, -2.0);

    let mut integrator: Box<dyn Integrator> = Box::new(SemiImplicitEuler);
    integrator.apply(&mut particles, 0.01).unwrap();

    let observer = KineticEnergyObserver::new(ParticleSelection::All);
    assert_eq!(observer.selection(), ParticleSelection::All);
    observer.observe(&particles).unwrap();

    let thermostat = LangevinThermostat::new(
        1.0,
        0.2,
        RngConfig::new(Some(29), None),
        ParticleSelection::AliveOnly,
    )
    .unwrap();
    assert_eq!(thermostat.tau_target(), 1.0);
    assert_eq!(thermostat.gamma(), 0.2);
    assert_eq!(thermostat.step_counter(), 0);
    assert_eq!(thermostat.selection(), ParticleSelection::AliveOnly);
}

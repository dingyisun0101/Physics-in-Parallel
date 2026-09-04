//! Actual callback participation and transactional custom-boundary regression.
use physics_in_parallel::prelude::models::*;
use physics_in_parallel::space::{BoundaryError, ContinuousBoundary};
use std::collections::HashSet;
use std::sync::Mutex;

struct Probe {
    workers: Mutex<HashSet<std::thread::ThreadId>>,
    fail: bool,
}
impl ContinuousBoundary for Probe {
    fn dim(&self) -> usize {
        1
    }
    fn apply_position(&self, position: &mut [f64]) -> Result<(), BoundaryError> {
        self.workers
            .lock()
            .unwrap()
            .insert(std::thread::current().id());
        position[0] += 1.0;
        if self.fail && position[0] > 1.0 {
            return Err(BoundaryError::InvalidVectorDimension {
                label: "probe",
                expected: 1,
                got: 2,
            });
        }
        Ok(())
    }
}
#[test]
fn thread_budget_and_custom_failure_are_observable_contracts() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    physics_in_parallel::threading::set_max_threads(Some(1)).unwrap();
    pool.install(|| {
        let mut particles = create_template(1, 20_000).unwrap();
        let probe = Probe {
            workers: Mutex::new(HashSet::new()),
            fail: false,
        };
        probe.apply_to_particles(&mut particles).unwrap();
        assert_eq!(probe.workers.lock().unwrap().len(), 1);
        let failing = Probe {
            workers: Mutex::new(HashSet::new()),
            fail: true,
        };
        particles
            .attribute_mut::<f64>(ATTR_R)
            .unwrap()
            .set(0, 0, 0.0)
            .unwrap();
        let before = particles
            .attribute::<f64>(ATTR_R)
            .unwrap()
            .values()
            .collect::<Vec<_>>();
        assert!(failing.apply_to_particles(&mut particles).is_err());
        assert_eq!(
            particles
                .attribute::<f64>(ATTR_R)
                .unwrap()
                .values()
                .collect::<Vec<_>>(),
            before
        );
    });
    physics_in_parallel::threading::set_max_threads(None).unwrap();
}

#[test]
fn indexed_sampling_and_pairs_replay_across_caller_pools() {
    use physics_in_parallel::prelude::basic::*;
    // This test does not mutate the global cap; the other test may cap jobs,
    // which must not affect the coordinate-to-sample mapping.
    let rng = ResolvedRng::new(42, RngMethod::IndexedSplitMix64);
    let mut reference = None;
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let result = pool.install(|| {
            let filler = TensorRandFiller::new(
                RandType::Uniform {
                    low: -1.0,
                    high: 2.0,
                },
                rng,
            )
            .unwrap();
            let mut values = Tensor::zeros(&[20_003], Backend::Dense).unwrap();
            filler.fill_at(&mut values, 8, 9).unwrap();
            let values: Vec<f64> = values.values().collect();
            let mut pairs =
                PairGenerator::new(&[7, 8, 9], PairingMethod::IndependentUniform, 10_003, rng)
                    .unwrap();
            pairs.refresh_at(11);
            (values, pairs.targets().values().collect::<Vec<_>>())
        });
        if let Some(reference) = &reference {
            assert_eq!(reference, &result);
        } else {
            reference = Some(result);
        }
    }
}

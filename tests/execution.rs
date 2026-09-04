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

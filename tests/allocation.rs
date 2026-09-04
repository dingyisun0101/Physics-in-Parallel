//! Allocation regression for the reusable dense public operations.
use physics_in_parallel::math::{Backend, Matrix, Tensor};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}
fn count() {
    let _ = ACTIVE.try_with(|active| {
        if active.get() {
            let _ = COUNT.try_with(|count| count.set(count.get() + 1));
        }
    });
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: forwarding the allocator's original layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count();
        // SAFETY: forwarding the original valid allocation and requested size.
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: forwarding the matching allocator and layout.
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn warmed_dense_outputs_do_not_allocate() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let a = Tensor::filled(&[100_003], Backend::Dense, 2.0_f64).unwrap();
        let b = a.clone();
        let mut out = a.clone();
        let matrix = Matrix::filled(5, 7, Backend::Dense, 2.0_f64).unwrap();
        let rhs = Matrix::filled(7, 3, Backend::Dense, 3.0_f64).unwrap();
        let mut product = Matrix::zeros(5, 3, Backend::Dense).unwrap();
        a.add_into(&b, &mut out).unwrap();
        matrix.matmul_into(&rhs, &mut product).unwrap();
        COUNT.set(0);
        ACTIVE.set(true);
        out.fill(1.0);
        out.map_in_place(|x| x + 1.0);
        a.add_into(&b, &mut out).unwrap();
        matrix.matmul_into(&rhs, &mut product).unwrap();
        ACTIVE.set(false);
        assert_eq!(COUNT.get(), 0);
        assert_eq!(product.get(0, 0).unwrap(), 42.0);
    });
}

#[test]
fn spatial_refresh_reuses_storage() {
    use physics_in_parallel::rng::{ResolvedRng, RngMethod};
    use physics_in_parallel::space::{
        ContinuousBoundary, KernelType, PairGenerator, PairingMethod, ReflectBox, SourceMode,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let boundary = ReflectBox::new(&[0.0; 3], &[1.0; 3]).unwrap();
        let mut positions = vec![2.25; 30_000];
        let mut velocities = vec![1.0; 30_000];
        boundary
            .apply_positions_velocities(&mut positions, &mut velocities)
            .unwrap();
        for method in [
            PairingMethod::IndependentUniform,
            PairingMethod::Kernel {
                kernel: KernelType::NearestNeighbor { dimension: 3 },
                sources: SourceMode::RandomUniform,
            },
            PairingMethod::Kernel {
                kernel: KernelType::UniformDistance { l: 3.0, c: 1.0 },
                sources: SourceMode::RandomUniform,
            },
        ] {
            let mut generator = PairGenerator::new(
                &[8, 9, 10],
                method,
                10_000,
                ResolvedRng::new(3, RngMethod::IndexedSplitMix64),
            )
            .unwrap();
            generator.refresh_at(7);
            let expected: Vec<_> = generator.targets().values().collect();
            COUNT.set(0);
            ACTIVE.set(true);
            generator.refresh_at(7);
            boundary
                .apply_positions_velocities(&mut positions, &mut velocities)
                .unwrap();
            ACTIVE.set(false);
            assert_eq!(COUNT.get(), 0, "{method:?}");
            assert_eq!(generator.targets().values().collect::<Vec<_>>(), expected);
            for ((source, displacement), target) in generator
                .sources()
                .values()
                .zip(generator.displacements().values())
                .zip(generator.targets().values())
            {
                assert_eq!(source + displacement, target);
            }
        }
    });
}

#[test]
fn dense_particle_step_and_observation_borrow_columns() {
    use physics_in_parallel::prelude::models::*;
    use physics_in_parallel::rng::{ResolvedRng, RngMethod};
    use physics_in_parallel::space::ReflectBox;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let mut particles = create_template(3, 10_000).unwrap();
        let boundary = ReflectBox::new(&[0.0; 3], &[1.0; 3]).unwrap();
        let mut thermostat = LangevinThermostat::new(
            1.0,
            0.5,
            ResolvedRng::new(5, RngMethod::IndexedSplitMix64),
            ParticleSelection::AliveOnly,
        )
        .unwrap();
        ExplicitEuler.apply(&mut particles, 0.01).unwrap();
        thermostat.apply(&mut particles, 0.01).unwrap();
        COUNT.set(0);
        ACTIVE.set(true);
        ExplicitEuler.apply(&mut particles, 0.01).unwrap();
        SemiImplicitEuler.apply(&mut particles, 0.01).unwrap();
        thermostat.apply(&mut particles, 0.01).unwrap();
        boundary.apply_to_particles(&mut particles).unwrap();
        let result = kinetic_summary(&particles, ParticleSelection::AliveOnly).unwrap();
        set_mass(&mut particles, 0, 2.0).unwrap();
        ACTIVE.set(false);
        assert_eq!(COUNT.get(), 0);
        assert!(result.energy.is_finite());
    });
}

#[test]
fn force_application_and_pair_replacement_do_not_allocate_dense_scratch() {
    use physics_in_parallel::prelude::models::*;
    let mut particles = create_template(3, 10_000).unwrap();
    let law = Spring::new(1.0, 1.0, None).unwrap();
    let mut springs = SpringNetwork::new();
    springs
        .insert_many((0..9999).map(|i| ((i, i + 1), law)))
        .unwrap();
    springs
        .apply(&mut particles, ParticleSelection::All)
        .unwrap();
    COUNT.set(0);
    ACTIVE.set(true);
    for _ in 0..100 {
        springs.insert((1, 0), law).unwrap();
        assert!(springs.get((0, 1)).is_some());
    }
    springs
        .apply(&mut particles, ParticleSelection::All)
        .unwrap();
    ACTIVE.set(false);
    assert_eq!(COUNT.get(), 0);
}

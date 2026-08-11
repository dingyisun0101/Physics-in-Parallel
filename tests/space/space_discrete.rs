use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::{ArrayD, IxDyn};

use physics_in_parallel::math::prelude::VectorList;
use physics_in_parallel::space::{
    discrete::square_lattice::{
        BoundaryCondition, Kernel, KernelType, NearestNeighborKernel, PairGenerationError,
        PairRandomKey, PowerLawKernel, RandPairGenerator, SourceMode, SquareLattice,
        SquareLatticeConfig, SquareLatticeInitMethod, UniformKernel, create_kernel,
    },
    io::square_lattice::save_square_lattice,
    space_trait::Space,
};

fn unique_tmp_json(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    env::temp_dir().join(format!("pip_{name}_{}_{}.json", std::process::id(), nanos))
}

#[test]
fn lattice_config_and_init_public_surface() {
    let cfg = SquareLatticeConfig::new(&vec![4; 2], BoundaryCondition::Periodic);
    assert_eq!(cfg.shape(), [4, 4]);
    assert_eq!(cfg.tensor_shape(), vec![4, 4]);
    assert_eq!(cfg.num_sites(), 16);

    let empty = SquareLattice::<usize>::new(cfg.clone(), SquareLatticeInitMethod::Empty);
    assert!(empty.data().iter().all(|&x| x == 0));

    let uniform =
        SquareLattice::<usize>::new(cfg.clone(), SquareLatticeInitMethod::Uniform { val: 3 });
    assert!(uniform.data().iter().all(|&x| x == 3));

    let random = SquareLattice::<usize>::new(
        cfg.clone(),
        SquareLatticeInitMethod::RandomUniformChoices {
            choices: vec![11, 22, 33],
        },
    );
    assert!(random.data().iter().all(|x| [11, 22, 33].contains(x)));

    let cfg_1d = SquareLatticeConfig::new(&vec![7; 1], BoundaryCondition::Reflective);
    let seeded = SquareLattice::<usize>::new(
        cfg_1d.clone(),
        SquareLatticeInitMethod::SeededCenter { val: 9 },
    );
    assert_eq!(seeded.data()[cfg_1d.shape()[0] / 2], 9);
}

#[test]
fn lattice_space_trait_vacancy_boundary_and_rescale_surface() {
    let cfg = SquareLatticeConfig::new(&vec![5; 1], BoundaryCondition::Periodic);
    let mut lattice = SquareLattice::<usize>::new(cfg, SquareLatticeInitMethod::Uniform { val: 1 });

    assert_eq!(Space::dims(&lattice), vec![5]);
    assert_eq!(Space::linear_size(&lattice), 5);
    assert_eq!(Space::data(&lattice).len(), 5);

    Space::set(&mut lattice, &[-1], 7);
    assert_eq!(*Space::get(&lattice, &[4]), 7);
    *Space::get_mut(&mut lattice, &[4]) = 8;
    assert_eq!(*Space::get(&lattice, &[-1]), 8);

    Space::set_all(&mut lattice, 2);
    assert!(lattice.data().iter().all(|&x| x == 2));

    assert_eq!(SquareLattice::<usize>::vacancy(), 0);
    lattice.set_vacant(&[2]);
    assert!(lattice.is_vacant(&[2]));
    lattice.fill_vacancy();
    assert!(lattice.data().iter().all(|&x| x == 0));

    let mut reflective = SquareLattice::<usize>::new(
        SquareLatticeConfig::new(&vec![5; 1], BoundaryCondition::Reflective),
        SquareLatticeInitMethod::Uniform { val: 1 },
    );
    Space::set(&mut reflective, &[-1], 77);
    assert_eq!(*Space::get(&reflective, &[1]), 77);
    Space::set(&mut reflective, &[5], 88);
    assert_eq!(*Space::get(&reflective, &[3]), 88);

    let lattice_2d = SquareLattice::<usize>::new(
        SquareLatticeConfig::new(&vec![4; 2], BoundaryCondition::Periodic),
        SquareLatticeInitMethod::Uniform { val: 5 },
    );
    let small = lattice_2d.rescale(&vec![2; lattice_2d.cfg.rank()]);
    assert_eq!(small.cfg.shape(), [2, 2]);
    assert_eq!(small.data().len(), 4);

    let clone = lattice_2d.rescale(&vec![4; lattice_2d.cfg.rank()]);
    assert_eq!(clone.cfg.shape(), [4, 4]);
    assert_eq!(clone.data(), lattice_2d.data());
}

#[test]
fn lattice_ndarray_and_save_surface() {
    let arr = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1usize, 2, 3, 4])
        .expect("ndarray shape should match data length");

    let lattice = SquareLattice::<usize>::from_ndarray(&arr, BoundaryCondition::Periodic);
    assert_eq!(lattice.cfg.shape(), [2, 2]);
    assert_eq!(lattice.to_ndarray(), arr);

    let out_1 = unique_tmp_json("save_square_lattice_fn");
    save_square_lattice(&lattice, &vec![2; lattice.cfg.rank()], &out_1)
        .expect("save_square_lattice should write json");
    let raw_1 = fs::read_to_string(&out_1).expect("saved json should be readable");
    assert!(raw_1.contains("\"shape\""));
    assert!(raw_1.contains("\"data\""));
    fs::remove_file(&out_1).expect("cleanup for save_square_lattice output should succeed");

    let out_2 = unique_tmp_json("space_save");
    Space::save(&lattice, &out_2, 2).expect("Space::save should write json");
    let raw_2 = fs::read_to_string(&out_2).expect("saved json should be readable");
    assert!(raw_2.contains("\"shape\""));
    fs::remove_file(&out_2).expect("cleanup for Space::save output should succeed");
}

#[test]
fn kernel_public_surface() {
    let key = PairRandomKey::new(91);
    let p = PowerLawKernel::new(10.0, 1.0, 2.0);
    assert!(matches!(p.kind(), KernelType::PowerLaw { .. }));
    let ps = p.sample_batch_indexed(key, 7, 64);
    assert_eq!(ps.len(), 64);
    assert!(ps.iter().all(|x| x.is_finite()));

    let u = UniformKernel::new(5.0, 2.0);
    assert!(matches!(u.kind(), KernelType::Uniform { .. }));
    let us = u.sample_batch_indexed(key, 7, 64);
    assert_eq!(us.len(), 64);
    assert!(us.iter().all(|x| *x >= 2.0 && *x < 5.0));

    let n = NearestNeighborKernel::new(3);
    assert!(matches!(n.kind(), KernelType::NearestNeighbor { .. }));
    let ns = n.sample_batch_indexed(key, 7, 64);
    assert_eq!(ns.len(), 64);
    assert!(ns.iter().all(|x| *x >= 0.0 && *x < 6.0));

    let k: Box<dyn Kernel> = create_kernel(KernelType::Uniform { l: 9.0, c: 1.5 });
    assert!(matches!(k.kind(), KernelType::Uniform { .. }));
    let k_clone = k.clone();
    assert_eq!(k.sample_batch_indexed(key, 7, 16).len(), 16);
    assert_eq!(
        k.sample_batch_indexed(key, 7, 16),
        k_clone.sample_batch_indexed(key, 7, 16)
    );
}

#[test]
fn rand_pair_generator_public_surface() {
    let mut nn_gen = RandPairGenerator::new(
        &[5, 7],
        KernelType::NearestNeighbor { d: 2 },
        32,
        SourceMode::RandomUniform,
        PairRandomKey::new(123),
    )
    .expect("valid nearest-neighbor pair generator");

    nn_gen.refresh_at(4);
    let src: &VectorList<isize> = nn_gen.sources();
    let disp: &VectorList<isize> = nn_gen.displacements();
    let tgt: &VectorList<isize> = nn_gen.targets();

    assert_eq!(src.shape(), [32, 2]);
    assert_eq!(disp.shape(), [32, 2]);
    assert_eq!(tgt.shape(), [32, 2]);
    assert_eq!(nn_gen.shape(), [5, 7]);
    assert_eq!(nn_gen.rank(), 2);
    assert_eq!(nn_gen.num_pairs(), 32);

    for i in 0..32 {
        assert!((0..5).contains(&src.get(i as isize, 0)));
        assert!((0..7).contains(&src.get(i as isize, 1)));

        let dx = disp.get(i as isize, 0);
        let dy = disp.get(i as isize, 1);
        let l1 = dx.abs() + dy.abs();
        assert_eq!(l1, 1, "nearest-neighbor displacement must be one-hot +/-1");
        assert_eq!(tgt.get(i as isize, 0), src.get(i as isize, 0) + dx);
        assert_eq!(tgt.get(i as isize, 1), src.get(i as isize, 1) + dy);
    }

    let mut pl_gen = RandPairGenerator::new(
        &[4, 5, 6],
        KernelType::PowerLaw {
            l: 20.0,
            c: 1.0,
            mu: 2.0,
        },
        16,
        SourceMode::Origin,
        PairRandomKey::new(456),
    )
    .expect("valid power-law pair generator");

    pl_gen.refresh_at(4);
    assert_eq!(pl_gen.sources().shape(), [16, 3]);
    assert_eq!(pl_gen.displacements().shape(), [16, 3]);
    assert_eq!(pl_gen.targets().shape(), [16, 3]);
    for i in 0..16 {
        for axis in 0..3 {
            assert_eq!(
                pl_gen.sources().get(i, axis),
                0,
                "with source filler = None, sources should remain at default zeros"
            );
        }
    }
}

#[test]
fn pair_generation_is_independent_of_rayon_worker_count() {
    fn flatten(vectors: &VectorList<isize>) -> Vec<isize> {
        let mut values = Vec::with_capacity(vectors.num_vectors() * vectors.dim());
        for pair in 0..vectors.num_vectors() {
            for axis in 0..vectors.dim() {
                values.push(vectors.get(pair as isize, axis as isize));
            }
        }
        values
    }

    fn generate(worker_count: usize) -> (Vec<isize>, Vec<isize>, Vec<isize>) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .expect("test Rayon pool should build")
            .install(|| {
                let mut generator = RandPairGenerator::new(
                    &[17, 19, 23],
                    KernelType::PowerLaw {
                        l: 12.0,
                        c: 1.0,
                        mu: 1.7,
                    },
                    1_024,
                    SourceMode::RandomUniform,
                    PairRandomKey::new(0x5eed),
                )
                .expect("valid indexed pair generator");
                generator.refresh_at(37);
                (
                    flatten(generator.sources()),
                    flatten(generator.displacements()),
                    flatten(generator.targets()),
                )
            })
    }

    assert_eq!(generate(1), generate(4));
}

#[test]
fn pair_generation_replays_an_explicit_sweep_exactly() {
    fn flatten(vectors: &VectorList<isize>) -> Vec<isize> {
        let mut values = Vec::with_capacity(vectors.num_vectors() * vectors.dim());
        for pair in 0..vectors.num_vectors() {
            for axis in 0..vectors.dim() {
                values.push(vectors.get(pair as isize, axis as isize));
            }
        }
        values
    }

    let mut generator = RandPairGenerator::new(
        &[11, 13],
        KernelType::NearestNeighbor { d: 2 },
        256,
        SourceMode::RandomUniform,
        PairRandomKey::new(44),
    )
    .expect("valid indexed pair generator");

    generator.refresh_at(9);
    let first = (
        flatten(generator.sources()),
        flatten(generator.displacements()),
        flatten(generator.targets()),
    );
    generator.refresh_at(10);
    let second_sources = flatten(generator.sources());
    assert_ne!(first.0, second_sources);
    generator.refresh_at(9);
    assert_eq!(generator.generated_sweep(), Some(9));
    assert_eq!(first.0, flatten(generator.sources()));
    assert_eq!(first.1, flatten(generator.displacements()));
    assert_eq!(first.2, flatten(generator.targets()));
}

#[test]
fn pair_generator_rejects_invalid_configuration_without_panicking() {
    let error = RandPairGenerator::new(
        &[4, 0],
        KernelType::NearestNeighbor { d: 2 },
        8,
        SourceMode::Origin,
        PairRandomKey::new(1),
    )
    .unwrap_err();
    assert_eq!(error, PairGenerationError::ZeroAxis { axis: 1 });

    let error = RandPairGenerator::new(
        &[4, 4],
        KernelType::NearestNeighbor { d: 1 },
        8,
        SourceMode::Origin,
        PairRandomKey::new(1),
    )
    .unwrap_err();
    assert_eq!(
        error,
        PairGenerationError::KernelRankMismatch {
            kernel_dimension: 1,
            rank: 2,
        }
    );
}

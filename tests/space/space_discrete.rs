use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::{ArrayD, IxDyn};

use physics_in_parallel::math::prelude::{RandType, TensorRandFiller, VectorList};
use physics_in_parallel::space::{
    discrete::{
        displacement::RandPairGenerator,
        representation::{save_grid, Grid, GridConfig, GridInitMethod},
    },
    kernel::{
        create_kernel, Kernel, KernelType, NearestNeighborKernel, PowerLawKernel, UniformKernel,
    },
    space_trait::Space,
};

/// Annotation:
/// - Purpose: Executes `unique_tmp_json` logic.
/// - Parameters:
///   - `name` (`&str`): Parameter of type `&str` used by `unique_tmp_json`.
fn unique_tmp_json(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    env::temp_dir().join(format!("pip_{name}_{}_{}.json", std::process::id(), nanos))
}

#[test]
/// Annotation:
/// - Purpose: Executes `grid_config_and_init_public_surface` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn grid_config_and_init_public_surface() {
    let cfg = GridConfig::new(2, 4, true);
    assert_eq!(cfg.shape(), [2, 4]);
    assert_eq!(cfg.num_sites(), 16);

    let g_empty = Grid::<usize>::new(cfg.clone(), GridInitMethod::Empty);
    assert!(g_empty.data.iter().all(|&x| x == 0));

    let g_uniform = Grid::<usize>::new(cfg.clone(), GridInitMethod::Uniform { val: 3 });
    assert!(g_uniform.data.iter().all(|&x| x == 3));

    let g_random = Grid::<usize>::new(
        cfg.clone(),
        GridInitMethod::RandomUniformChoices {
            choices: vec![11, 22, 33],
        },
    );
    assert!(g_random.data.iter().all(|x| [11, 22, 33].contains(x)));

    let cfg_1d = GridConfig::new(1, 7, false);
    let g_seeded = Grid::<usize>::new(cfg_1d.clone(), GridInitMethod::SeededCenter { val: 9 });
    assert_eq!(g_seeded.data[cfg_1d.l / 2], 9);
}

#[test]
/// Annotation:
/// - Purpose: Executes `grid_space_trait_vacancy_and_rescale_surface` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn grid_space_trait_vacancy_and_rescale_surface() {
    let cfg = GridConfig::new(1, 5, true);
    let mut g = Grid::<usize>::new(cfg, GridInitMethod::Uniform { val: 1 });

    assert_eq!(Space::dims(&g), vec![1, 5]);
    assert_eq!(Space::linear_size(&g), 5);
    assert_eq!(Space::data(&g).len(), 5);

    Space::set(&mut g, &[-1], 7);
    assert_eq!(*Space::get(&g, &[4]), 7);
    *Space::get_mut(&mut g, &[4]) = 8;
    assert_eq!(*Space::get(&g, &[-1]), 8);

    Space::set_all(&mut g, 2);
    assert!(g.data.iter().all(|&x| x == 2));

    assert_eq!(Grid::<usize>::vacancy(), 0);
    g.set_vacant(&[2]);
    assert!(g.is_vacant(&[2]));
    g.fill_vacancy();
    assert!(g.data.iter().all(|&x| x == 0));

    let mut g_non_periodic = Grid::<usize>::new(
        GridConfig::new(1, 5, false),
        GridInitMethod::Uniform { val: 1 },
    );
    Space::set(&mut g_non_periodic, &[-9], 77);
    assert_eq!(*Space::get(&g_non_periodic, &[0]), 77);

    let g2 = Grid::<usize>::new(
        GridConfig::new(2, 4, true),
        GridInitMethod::Uniform { val: 5 },
    );
    let g2_small = g2.rescale(2);
    assert_eq!(g2_small.cfg.shape(), [2, 2]);
    assert_eq!(g2_small.data.len(), 4);

    let g2_clone = g2.rescale(8);
    assert_eq!(g2_clone.cfg.l, 4);
    assert_eq!(g2_clone.data, g2.data);
}

#[test]
/// Annotation:
/// - Purpose: Executes `grid_ndarray_and_save_surface` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn grid_ndarray_and_save_surface() {
    let arr = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1usize, 2, 3, 4])
        .expect("ndarray shape should match data length");

    let g = Grid::<usize>::from_ndarry(&arr, true);
    assert_eq!(g.cfg.d, 2);
    assert_eq!(g.cfg.l, 2);
    assert_eq!(g.to_ndarray(), arr);

    let out_1 = unique_tmp_json("save_grid_fn");
    save_grid(&g, 2, &out_1).expect("save_grid should write json");
    let raw_1 = fs::read_to_string(&out_1).expect("saved json should be readable");
    assert!(raw_1.contains("\"shape\""));
    assert!(raw_1.contains("\"data\""));
    fs::remove_file(&out_1).expect("cleanup for save_grid output should succeed");

    let out_2 = unique_tmp_json("space_save");
    Space::save(&g, &out_2, 2).expect("Space::save should write json");
    let raw_2 = fs::read_to_string(&out_2).expect("saved json should be readable");
    assert!(raw_2.contains("\"shape\""));
    fs::remove_file(&out_2).expect("cleanup for Space::save output should succeed");
}

#[test]
/// Annotation:
/// - Purpose: Executes `kernel_public_surface` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn kernel_public_surface() {
    let p = PowerLawKernel::new(10.0, 1.0, 2.0);
    assert!(matches!(p.kind(), KernelType::PowerLaw { .. }));
    let ps = p.sample(64);
    assert_eq!(ps.len(), 64);
    assert!(ps.iter().all(|x| x.is_finite()));

    let u = UniformKernel::new(5.0, 2.0);
    assert!(matches!(u.kind(), KernelType::Uniform { .. }));
    let us = u.sample(64);
    assert_eq!(us.len(), 64);
    assert!(us.iter().all(|x| *x >= 2.0 && *x < 5.0));

    let n = NearestNeighborKernel::new(3);
    assert!(matches!(n.kind(), KernelType::NearestNeighbor { .. }));
    let ns = n.sample(64);
    assert_eq!(ns.len(), 64);
    assert!(ns.iter().all(|x| *x >= 0.0 && *x < 6.0));

    let k: Box<dyn Kernel> = create_kernel(KernelType::Uniform { l: 9.0, c: 1.5 });
    assert!(matches!(k.kind(), KernelType::Uniform { .. }));
    let k_clone = k.clone();
    assert_eq!(k.sample(16).len(), 16);
    assert_eq!(k_clone.sample(16).len(), 16);
}

#[test]
/// Annotation:
/// - Purpose: Executes `rand_pair_generator_public_surface` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn rand_pair_generator_public_surface() {
    let filler = TensorRandFiller::new(RandType::UniformInt { low: -5, high: 5 }, Some(2));
    let mut nn_gen = RandPairGenerator::new(
        KernelType::NearestNeighbor { d: 2 },
        2,
        32,
        Some(filler),
        Some(2),
    );

    nn_gen.refresh();
    let src: &VectorList<isize> = nn_gen.sources();
    let tgt: &VectorList<isize> = nn_gen.targets();

    assert_eq!(src.shape(), [32, 2]);
    assert_eq!(tgt.shape(), [32, 2]);

    for i in 0..32 {
        let dx = tgt.get(i as isize, 0) - src.get(i as isize, 0);
        let dy = tgt.get(i as isize, 1) - src.get(i as isize, 1);
        let l1 = dx.abs() + dy.abs();
        assert_eq!(l1, 1, "nearest-neighbor displacement must be one-hot ±1");
    }

    let mut pl_gen = RandPairGenerator::new(
        KernelType::PowerLaw {
            l: 20.0,
            c: 1.0,
            mu: 2.0,
        },
        3,
        16,
        None,
        Some(2),
    );

    pl_gen.refresh();
    assert_eq!(pl_gen.sources().shape(), [16, 3]);
    assert_eq!(pl_gen.targets().shape(), [16, 3]);
    assert!(
        pl_gen.sources().as_tensor().data.iter().all(|&x| x == 0),
        "with source filler = None, sources should remain at default zeros"
    );
}

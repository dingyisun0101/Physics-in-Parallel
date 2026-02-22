

use math::scalar::Scalar;
use space::space_trait::Space;
use space::discrete::representation::{Grid, GridConfig, GridInitMethod};
use space::discrete::displacement::RandPairGenerator;
use space::kernel::KernelType;

fn vec_to_string <T:Scalar> (v: &Vec<T>) -> String {
    let vec_str = v.iter()
    .map(|x| x.to_string())
    .collect::<Vec<_>>()
    .join(", ");
    vec_str
}

pub fn test_discrete_1d() {
    let cfg = GridConfig { d: 1, l: 50, periodic: (true) };
    let mut space: Grid<usize> = Grid::new(cfg.clone(), GridInitMethod::RandomUniformChoices { choices: (vec![1,2,3]) });
    
    let data_str = vec_to_string(&space.data);
    println!("Initial Grid: {data_str}");

    let i = -5;
    let site_i = space.get(&[i]); 
    println!("Site {i}: {site_i}");

    space.set(&[i], 100 as usize);
    let data_str = vec_to_string(&space.data);
    println!("Current Grid: {data_str}");

    let mut rpg = RandPairGenerator::new(KernelType::PowerLaw { l: cfg.l as f64, c: 2.0, mu: 3.5 }, cfg.d, 10);
    rpg.random(true);
    
    let source_str = rpg.source_coords_cache.data.to_string();
    let target_str = rpg.target_coords_cache.data.to_string();

    println!("Sources: {source_str}");
    println!("Targets: {target_str}");
}

pub fn test_discrete_2d() {
    let cfg = GridConfig { d: 2, l: 20, periodic: (true) };
    let mut space: Grid<usize> = Grid::new(cfg.clone(), GridInitMethod::RandomUniformChoices { choices: (vec![1,2,3]) });
    
    let data_str = vec_to_string(&space.data);
    println!("Initial Grid: {data_str}");

    let i = [-5, 4];
    let site_i = space.get(&i); 
    println!("Site {:?}: {}", i, site_i);

    space.set(&i, 100 as usize);
    let data_str = vec_to_string(&space.data);
    println!("Current Grid: {data_str}");

    let mut rpg = RandPairGenerator::new(KernelType::PowerLaw { l: cfg.l as f64, c: 1.5, mu: 100.0 }, cfg.d, 10);
    rpg.random(true);
    
    let source_str = rpg.source_coords_cache.data.to_string();
    let target_str = rpg.target_coords_cache.data.to_string();

    println!("Sources: {source_str}");
    println!("Targets: {target_str}");

    let j = 5;
    let vec_j = rpg.source_coords_cache.get_vec(j);
    print!("vec_j: {}", vec_to_string(&vec_j));


}

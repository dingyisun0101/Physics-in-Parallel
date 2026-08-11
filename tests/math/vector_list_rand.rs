use ndarray::arr2;
use physics_in_parallel::math::tensor::rank_2::vector_list::{
    HaarVectors, NNVectors, VectorListRand,
};
use physics_in_parallel::rng::RngConfig;

#[test]
fn haar_vectors_refresh_generates_unit_vectors_in_parallel_storage() {
    let dim = 3;
    let n = 16;
    let mut generator = HaarVectors::new(dim, n, RngConfig::default()).unwrap();
    generator.refresh();

    assert_eq!(generator.dim, dim);
    assert_eq!(generator.n, n);
    assert_eq!(generator.vl.shape(), [n, dim]);

    let mut component_sum = vec![0.0; dim];
    for i in 0..n {
        let row = generator.vl.get_vec(i as isize);
        assert_eq!(row.len(), dim);

        let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "Haar vector norm was {norm}");

        for (axis, x) in row.iter().copied().enumerate() {
            component_sum[axis] += x;
        }
    }

    for norm in generator.vl.norms_real() {
        assert!((norm - 1.0).abs() < 1e-10);
    }

    println!("\nfirst refreshed Haar vectors");
    for i in 0..3 {
        println!("{:?}", generator.vl.get_vec(i));
    }
    println!("Haar component sum over {n} vectors: {component_sum:?}");
}

#[test]
fn nearest_neighbor_vectors_refresh_generates_one_hot_signed_rows() {
    let mut generator = NNVectors::new(4, 32, RngConfig::default()).unwrap();
    generator.refresh();

    assert_eq!(generator.vl.shape(), [32, 4]);
    for i in 0..generator.vl.num_vecs() {
        let row = generator.vl.get_vec(i as isize);
        let nonzero: Vec<_> = row.iter().copied().filter(|&x| x != 0).collect();
        assert_eq!(nonzero.len(), 1);
        assert!(nonzero[0] == 1 || nonzero[0] == -1);
    }

    println!("\nfirst refreshed nearest-neighbor vectors");
    for i in 0..4 {
        println!("{:?}", generator.vl.get_vec(i));
    }
}

#[test]
fn vector_list_random_wrappers_preserve_ndarray_interop() {
    let haar_array = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let haar = HaarVectors::from_ndarray(&haar_array, RngConfig::default()).unwrap();
    assert_eq!(haar.dim, 2);
    assert_eq!(haar.n, 2);
    assert_eq!(haar.to_ndarray(), haar_array);

    let nn_array = arr2(&[[1isize, 0isize], [0isize, -1isize]]);
    let nn = NNVectors::from_ndarray(&nn_array, RngConfig::default()).unwrap();
    assert_eq!(nn.dim, 2);
    assert_eq!(nn.n, 2);
    assert_eq!(nn.to_ndarray(), nn_array);
}

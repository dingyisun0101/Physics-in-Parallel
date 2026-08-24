use physics_in_parallel::prelude::advanced::{HaarVectors, NNVectors, VectorListRand};
use physics_in_parallel::prelude::basic::RngConfig;

fn snapshot<T>(rows: &physics_in_parallel::prelude::basic::VectorList<T>) -> Vec<T>
where
    T: Copy + physics_in_parallel::prelude::basic::Scalar,
{
    (0..rows.num_vectors())
        .flat_map(|index| rows.vector(index as isize).iter().copied())
        .collect()
}

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
        let row = generator.vl.vector(i as isize);
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
        println!("{:?}", generator.vl.vector(i));
    }
    println!("Haar component sum over {n} vectors: {component_sum:?}");
}

#[test]
fn indexed_haar_vectors_specialize_low_dimensions_and_replay_steps() {
    for dim in 1..=4 {
        let mut generator = HaarVectors::try_new_indexed(dim, 256, RngConfig::default()).unwrap();
        generator.try_refresh_at(7, 0x4841_4152).unwrap();
        let first = snapshot(&generator.vl);
        generator.try_refresh_at(8, 0x4841_4152).unwrap();
        assert_ne!(snapshot(&generator.vl), first);
        generator.try_refresh_at(7, 0x4841_4152).unwrap();
        assert_eq!(snapshot(&generator.vl), first);

        for row in first.chunks_exact(dim) {
            let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-10);
            if dim == 1 {
                assert!(row[0] == -1.0 || row[0] == 1.0);
            }
        }
    }
}

#[test]
fn indexed_nearest_neighbor_vectors_are_exact_and_replay_steps() {
    let mut generator = NNVectors::try_new_indexed(5, 512, RngConfig::default()).unwrap();
    generator.try_refresh_at(11, 0x4e4e).unwrap();
    let first = snapshot(&generator.vl);
    generator.try_refresh_at(12, 0x4e4e).unwrap();
    assert_ne!(snapshot(&generator.vl), first);
    generator.try_refresh_at(11, 0x4e4e).unwrap();
    assert_eq!(snapshot(&generator.vl), first);

    for row in first.chunks_exact(5) {
        assert_eq!(row.iter().filter(|&&value| value != 0).count(), 1);
        assert!(row.iter().all(|value| (-1..=1).contains(value)));
    }
}

#[test]
fn nearest_neighbor_vectors_refresh_generates_one_hot_signed_rows() {
    let mut generator = NNVectors::new(4, 32, RngConfig::default()).unwrap();
    generator.refresh();

    assert_eq!(generator.vl.shape(), [32, 4]);
    for i in 0..generator.vl.num_vectors() {
        let row = generator.vl.vector(i as isize);
        let nonzero: Vec<_> = row.iter().copied().filter(|&x| x != 0).collect();
        assert_eq!(nonzero.len(), 1);
        assert!(nonzero[0] == 1 || nonzero[0] == -1);
    }

    println!("\nfirst refreshed nearest-neighbor vectors");
    for i in 0..4 {
        println!("{:?}", generator.vl.vector(i));
    }
}

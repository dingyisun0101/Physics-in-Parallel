use physics_in_parallel::prelude::basic::{Scalar, VectorList};

fn print_vector_list<T>(label: &str, vectors: &VectorList<T>)
where
    T: Scalar + Copy,
{
    println!("\n{label}");
    vectors.print();
}

#[test]
fn vector_list_elementwise_and_scalar_ops_are_type_preserving() {
    let a = VectorList::<i64>::from_vec(3, 2, vec![2, 4, 6, 8, 10, 12]);
    let b = VectorList::<i64>::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);

    let sum = a.add(&b);
    assert_eq!(sum.vector(0), &[3, 6, 9]);
    assert_eq!(sum.vector(1), &[12, 15, 18]);
    print_vector_list("vector list sum", &sum);

    let diff = a.sub(&b);
    assert_eq!(diff.vector(0), &[1, 2, 3]);
    assert_eq!(diff.vector(1), &[4, 5, 6]);

    let elem_mul = a.elem_mul(&b);
    assert_eq!(elem_mul.vector(1), &[32, 50, 72]);

    let elem_div = a.elem_div(&b);
    assert_eq!(elem_div.vector(0), &[2, 2, 2]);
    assert_eq!(elem_div.vector(1), &[2, 2, 2]);

    let scaled = a.scalar_mul(3);
    assert_eq!(scaled.vector(0), &[6, 12, 18]);

    assert_eq!((&a + &b).vector(0), &[3, 6, 9]);
    assert_eq!((&a * 2).vector(1), &[16, 20, 24]);
}

#[test]
fn vector_list_parallel_vector_mutation_and_scaling_operate_rowwise() {
    let mut vectors = VectorList::<i64>::from_vec(3, 3, vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);

    vectors.par_for_each_vector_mut(|i, row| {
        for x in row {
            *x *= (i + 1) as i64;
        }
    });
    assert_eq!(vectors.vector(0), &[1, 1, 1]);
    assert_eq!(vectors.vector(1), &[4, 4, 4]);
    assert_eq!(vectors.vector(2), &[9, 9, 9]);

    vectors.scale_vectors_by_list(&[10, 20, 30]);
    assert_eq!(vectors.vector(0), &[10, 10, 10]);
    assert_eq!(vectors.vector(1), &[80, 80, 80]);
    assert_eq!(vectors.vector(2), &[270, 270, 270]);
    print_vector_list("rowwise vector mutation and scaling", &vectors);
}

#[test]
fn vector_list_norms_normalize_polar_and_cast_cover_conversion_boundary() {
    let mut vectors = VectorList::<f64>::from_vec(2, 3, vec![3.0, 4.0, 0.0, 0.0, 5.0, 12.0]);

    let norms = vectors.norms();
    assert_eq!(norms, vec![5.0, 0.0, 13.0]);
    assert_eq!(vectors.norms_real(), vec![5.0, 0.0, 13.0]);

    let (polar_norms, units) = vectors.to_polar();
    assert_eq!(polar_norms, vec![5.0, 0.0, 13.0]);
    assert!((units.get(0, 0) - 0.6).abs() < 1e-12);
    assert_eq!(units.vector(1), &[0.0, 0.0]);
    assert!((units.get(2, 1) - 12.0 / 13.0).abs() < 1e-12);

    vectors.normalize();
    assert!((vectors.get(0, 0) - 0.6).abs() < 1e-12);
    assert_eq!(vectors.vector(1), &[0.0, 0.0]);
    assert!((vectors.get(2, 1) - 12.0 / 13.0).abs() < 1e-12);
    print_vector_list("normalized vector list", &vectors);

    let ints = VectorList::<i64>::from_vec(2, 2, vec![1, 2, 3, 4]);
    let floats = ints
        .try_cast_to::<f64>()
        .expect("i64 to f64 cast should work");
    assert_eq!(floats.vector(1), &[3.0, 4.0]);
}

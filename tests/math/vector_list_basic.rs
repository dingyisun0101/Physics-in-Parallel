use physics_in_parallel::prelude::advanced::DynVectorList;
use physics_in_parallel::prelude::basic::{Scalar, VectorList};

fn print_vector_list<T>(label: &str, vectors: &VectorList<T>)
where
    T: Scalar + Copy,
{
    println!("\n{label}");
    vectors.print();
}

#[test]
fn vector_list_constructs_dense_rank_n_rows_and_exposes_vector_first_access() {
    let mut vectors = VectorList::<i64>::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);

    assert_eq!(vectors.shape(), [2, 3]);
    assert_eq!(vectors.num_vectors(), 2);
    assert_eq!(vectors.num_vectors(), 2);
    assert_eq!(vectors.dim(), 3);
    assert_eq!(vectors.vector(0), &[1, 2, 3]);
    assert_eq!(vectors.vector(-1), &[4, 5, 6]);
    assert_eq!(vectors.get(1, 2), 6);

    vectors.set_vector(0, &[7, 8, 9]);
    assert_eq!(vectors.vector(0), &[7, 8, 9]);

    vectors.vector_mut(-1)[1] = 50;
    assert_eq!(vectors.get(1, 1), 50);

    vectors.set(-1, -1, 60);
    assert_eq!(vectors.vector(1), &[4, 50, 60]);
    assert_eq!(vectors.vector_owned(1), vec![4, 50, 60]);
    print_vector_list("vector-first access", &vectors);
}

#[test]
fn vector_list_axis_fill_from_fn_and_slice_updates_are_consistent() {
    let mut vectors = VectorList::<i64>::from_fn(3, 4, |i, k| (10 * i + k) as i64);

    assert_eq!(vectors.vector(2), &[20, 21, 22]);
    assert_eq!(vectors.axis(1), vec![1, 11, 21, 31]);
    assert_eq!(vectors.axis(-1), vec![2, 12, 22, 32]);

    vectors.set_axis(1, &[100, 101, 102, 103]);
    assert_eq!(vectors.axis(1), vec![100, 101, 102, 103]);

    vectors.set_vector(-1, &[70, 71, 72]);
    assert_eq!(vectors.vector(3), &[70, 71, 72]);
    print_vector_list("axis and vector slice updates", &vectors);

    vectors.fill(5);
    assert_eq!(vectors.vector(0), &[5, 5, 5]);
    assert_eq!(vectors.vector(3), &[5, 5, 5]);
}

#[test]
fn vector_list_json_and_dynamic_trait_paths_preserve_values() {
    let vectors = VectorList::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    assert_eq!(vectors.shape(), [3, 2]);

    let json = vectors.serialize().expect("vector list should serialize");
    let decoded: VectorList<f64> =
        serde_json::from_str(&json).expect("vector list should deserialize");
    assert_eq!(decoded.vector(2), &[5.0, 6.0]);

    let boxed: Box<dyn DynVectorList> = Box::new(decoded.clone());
    assert_eq!(boxed.dim(), 2);
    assert_eq!(boxed.num_vectors(), 3);
    assert!(boxed.serialize_value().is_ok());

    let cloned = boxed.clone();
    let downcast = cloned
        .as_any()
        .downcast_ref::<VectorList<f64>>()
        .expect("dynamic vector list should downcast to original scalar type");
    assert_eq!(downcast.vector(1), &[3.0, 4.0]);
}

#[test]
/// Annotation:
/// - Purpose: Executes `vector_list` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn vector_list() {
        use physics_in_parallel::math::{
        tensor::rank_2::vector_list::VectorList,
    };
    
    // Make 3 vectors in R^3, all zeros initially.
    let mut vl: VectorList<f64> = VectorList::empty(3, 20);
    assert_eq!(vl.shape(), [3, 20]);

    // Set vectors:
    // v0 = (1, 2, 3), v1 = (4, 5, 6), v2 = (7, 8, 9)
    for (i, triple) in [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]].into_iter().enumerate() {
        for (k, val) in triple.into_iter().enumerate() {
            vl.set(i as isize, k as isize, val);
        }
    }

    // Immutable reads via safe accessors:
    assert_eq!(vl.get(0, 2), 3.0); // v0.z

    // Mutable view over a single vector:
    {
        let v1 = vl.get_vector_mut(1);
        // scale v1.y by 10
        *v1.get_mut(1).expect("index 1 must exist") *= 10.0;
    }
    assert_eq!(vl.get_vector(1).to_vec().as_ref(), [4.0, 50.0, 6.0]);

    // Decompose into norms and unit vectors:
    let (norms, mut units) = vl.to_polar();
    assert_eq!(units.shape(), [3, 20]);

    // Scale each unit vector by its norm (reconstruct original magnitudes).
    units.scale_vectors_by_list(&norms.clone());
    let vl_scaled_back = units; // rename for clarity
    // vl_scaled_back is approximately the pre-normalized vl (up to float error).

    // Elementwise arithmetic: delegate to Tensor’s parallel ops
    let _sum = &vl_scaled_back.clone() + &vl_scaled_back.clone();
    let _diff = &vl_scaled_back.clone() - &vl_scaled_back.clone();
    let _hadamard = &vl_scaled_back.clone() * &vl_scaled_back.clone();
}


#[test]
/// Annotation:
/// - Purpose: Executes `vector_list_rand` logic.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn vector_list_rand() {
        use physics_in_parallel::math::{
        tensor::rank_2::{ 
            vector_list_rand::{HaarVectors, NNVectors, VectorListRand},
        },
    };
    
    let n = 10;
    
    // 2d Haar vectors
    let mut hv_1 = HaarVectors::new(2, n, None);
    assert_eq!(hv_1.vl.shape(), [2, n]);
    hv_1.refresh();
    hv_1.vl.print();

    // 1d Haar vectors cast into isize
    let mut hv_2 = HaarVectors::new(1, n, None);
    assert_eq!(hv_2.vl.shape(), [1, n]);
    hv_2.refresh();
    let hv_2_vl = hv_2.vl.cast_to::<isize>();
    hv_2_vl.print();

    let mut nn_vectors = NNVectors::new(1, n, None);
    assert_eq!(nn_vectors.vl.shape(), [1, n]);
    nn_vectors.refresh();
    nn_vectors.vl.print();
}

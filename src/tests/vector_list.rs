#[test]
fn vector_list() {
        use physics_in_parallel::math::{
        tensor_2d::vector_list::VectorList,
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
        let mut v1 = vl.get_vector_mut(1);
        // scale v1.y by 10
        *v1.get_mut(1) *= 10.0;
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
fn vector_list_rand() {
        use physics_in_parallel::math::{
        tensor_2d::{ 
            vector_list_rand::{HaarVectors, NNVectors, VectorListRand},
        },
    };
    
    let n = 10;
    
    let mut haar_vectors = HaarVectors::new(2, n);
    assert_eq!(haar_vectors.vl.shape(), [2, n]);
    haar_vectors.refresh();
    haar_vectors.vl.print();

    let mut nn_vectors = NNVectors::new(1, n);
    assert_eq!(nn_vectors.vl.shape(), [1, n]);
    nn_vectors.refresh();
    nn_vectors.vl.print();
}

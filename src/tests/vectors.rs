
use physics_in_parallel::math_fundations::tensor::dense::Tensor;
use physics_in_parallel::math_fundations::{vector_list::VectorList};
// ============================================================================
// ----------------------------- Usage Example --------------------------------
// ============================================================================

/*
Run as a doc-test after adapting module paths & ensuring your `Scalar` trait
is implemented for `f64` (or for your own scalar wrapper).
*/

#[test]
fn vector_list_demo() {
    // Make 3 vectors in R^3, all zeros initially.
    let mut vl: VectorList<f64, 3> = VectorList::new(3);
    assert_eq!(vl.as_tensor().shape, vec![3, 3]);

    // Set vectors:
    // v0 = (1, 2, 3), v1 = (4, 5, 6), v2 = (7, 8, 9)
    for (i, triple) in [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]].into_iter().enumerate() {
        for (k, val) in triple.into_iter().enumerate() {
            vl.set(i, k, val);
        }
    }

    // Immutable reads via safe accessors:
    assert_eq!(*vl.get(0, 2), 3.0); // v0.z

    // Mutable view over a single vector:
    {
        let mut v1 = vl.get_vec_mut(1);
        // scale v1.y by 10
        *v1.get_mut(1) *= 10.0;
    }
    assert_eq!(vl.get_vec(1), [4.0, 50.0, 6.0]);

    // Normalize all vectors:
    vl.normalize();

    // Decompose into norms and unit vectors:
    let (norms, units) = vl.to_polar();
    assert_eq!(norms.shape, vec![3]);
    assert_eq!(units.as_tensor().shape, vec![3, 3]);

    // Scale each unit vector by its norm (reconstruct original magnitudes).
    let vl_scaled_back = units.scale_by_list(&norms.data.clone());
    // vl_scaled_back is approximately the pre-normalized vl (up to float error).

    // Elementwise arithmetic: delegate to Tensor’s parallel ops
    let _sum = vl_scaled_back.clone() + vl_scaled_back.clone();
    let _diff = vl_scaled_back.clone() - vl_scaled_back.clone();
    let _hadamard = vl_scaled_back.clone() * vl_scaled_back.clone();
}


fn test_haar_random_vectors() {
    use physics_in_parallel::math::{vector::vector_list::VectorList};
    use physics_in_parallel::math::{vector::vector_list_rand::fill_haar_vectors};
    
    let d = 3;
    let n = 20;

    // 1) Create empty vector list
    let mut vl = VectorList::<f64>::new(d, n);

    // 2) Fill with Haar-random unit vectors
    fill_haar_vectors(&mut vl);

    // 3) Print and check normalization
    println!("Generated {n} Haar random vectors in {d}D:");
    for i in 0..n {
        let v = vl.get_vec(i as isize);

        // compute L2 norm
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();

        println!("{:?}, norm = {:.6}", v, norm);

        // check ~1.0
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Vector {} not normalized: norm = {}",
            i,
            norm
        );
    }
}
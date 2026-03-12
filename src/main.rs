
/// Annotation:
/// - Purpose: Executes `main` logic for this module.
/// - Parameters:
///   - (none): This function has no documented non-receiver parameters.
///   - (none): This function takes no explicit parameters.
fn main() {}

#[test]
/// Annotation:
/// - Purpose: Executes `vector_list_rand` logic for this module.
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
    assert_eq!(hv_1.vl.shape(), [n, 2]);
    hv_1.refresh();
    hv_1.vl.print();

    // 1d Haar vectors cast into isize
    let mut hv_2 = HaarVectors::new(1, n, None);
    assert_eq!(hv_2.vl.shape(), [n, 1]);
    hv_2.refresh();
    let hv_2_vl = hv_2.vl.cast_to::<isize>();
    hv_2_vl.print();

    let mut nn_vectors = NNVectors::new(1, n, None);
    assert_eq!(nn_vectors.vl.shape(), [n, 1]);
    nn_vectors.refresh();
    nn_vectors.vl.print();
}

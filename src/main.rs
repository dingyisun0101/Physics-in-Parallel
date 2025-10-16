
fn main() {}

#[test]
fn vector_list_rand() {
        use physics_in_parallel::math::{
        tensor_2d::{ 
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

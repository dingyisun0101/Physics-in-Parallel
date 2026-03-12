use ndarray::arr2;
use physics_in_parallel::math::tensor::VectorList;

fn main() {
    // [n, dim] = [2, 3]
    let array = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let vectors = VectorList::<f64>::from_ndarray(&array);
    assert_eq!(vectors.shape(), [2, 3]);

    let back = vectors.to_ndarray();
    assert_eq!(back, array);

    println!("VectorList shape (n, dim): {:?}", vectors.shape());
    println!("Roundtrip ndarray:\n{back}");
}

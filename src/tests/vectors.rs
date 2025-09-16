
use physics_in_parallel::math_fundations::{vector_list::VectorList};


fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}
fn approx_eq_slice(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(a.len(), b.len(), "len mismatch: {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            approx_eq(*x, *y, eps),
            "idx {}: {} != {} (eps={})",
            i, x, y, eps
        );
    }
}

#[test]
fn accessors_and_num_vec() {
    let mut vl = VectorList::<f64, 3>::new(4);
    assert_eq!(vl.dim(), 3);
    assert_eq!(vl.num_vec(), 4);

    // Set a few rows via mutable vector proxy
    vl.get_vec_mut(0).set_from([1.0, 2.0, 3.0]);
    vl.get_vec_mut(1).set_from([0.5, -1.0, 4.0]);

    // Read back
    assert_eq!(vl.get_vec(0), [1.0, 2.0, 3.0]);
    assert_eq!(vl.get(1, 2), &4.0);

    // dim slices (SoA)
    approx_eq_slice(vl.dim_slice(0), &[1.0, 0.5, 0.0, 0.0], 1e-12);
    approx_eq_slice(vl.dim_slice(1), &[2.0, -1.0, 0.0, 0.0], 1e-12);
    approx_eq_slice(vl.dim_slice(2), &[3.0, 4.0, 0.0, 0.0], 1e-12);
}

#[test]
fn to_polar_and_normalize() {
    // 2D: rows are vectors (3,4), (0,0), (1,0)
    let mut vl = VectorList::<f64, 2>::new(3);
    vl.get_vec_mut(0).set_from([3.0, 4.0]);
    vl.get_vec_mut(1).set_from([0.0, 0.0]);
    vl.get_vec_mut(2).set_from([1.0, 0.0]);

    let (norms, units) = vl.to_polar();
    assert_eq!(norms.shape, vec![3]);
    assert!(approx_eq(norms.data[0], 5.0, 1e-12));
    assert!(approx_eq(norms.data[1], 0.0, 1e-12));
    assert!(approx_eq(norms.data[2], 1.0, 1e-12));

    let u0 = units.get_vec(0);
    assert!(approx_eq(u0[0], 0.6, 1e-12));
    assert!(approx_eq(u0[1], 0.8, 1e-12));

    // In-place normalization matches units
    vl.normalize();
    let v0 = vl.get_vec(0);
    let v1 = vl.get_vec(1);
    let v2 = vl.get_vec(2);
    approx_eq_slice(&v0, &u0, 1e-12);
    approx_eq_slice(&v1, &units.get_vec(1), 1e-12);
    approx_eq_slice(&v2, &units.get_vec(2), 1e-12);
}

#[test]
fn arithmetic_ops_delegate_to_tensor() {
    // Build two VectorLists with D=2, n=3
    let mut a = VectorList::<f64, 2>::new(3);
    let mut b = VectorList::<f64, 2>::new(3);

    a.get_vec_mut(0).set_from([1.0, 2.0]);
    a.get_vec_mut(1).set_from([3.0, 4.0]);
    a.get_vec_mut(2).set_from([5.0, 6.0]);

    b.get_vec_mut(0).set_from([10.0, 1.0]);
    b.get_vec_mut(1).set_from([2.0, 3.0]);
    b.get_vec_mut(2).set_from([4.0, 5.0]);

    let sum = a.clone() + b.clone();
    let diff = a.clone() - b.clone();
    let prod = a.clone() * b.clone();
    let quot = a.clone() / b.clone();

    // Check a few entries (remember SoA layout: dim k are contiguous)
    // i=1 (second vector)
    assert!(approx_eq(*sum.get(1, 0), 3.0 + 2.0, 1e-12));  // 5.0
    assert!(approx_eq(*sum.get(1, 1), 4.0 + 3.0, 1e-12));  // 7.0

    assert!(approx_eq(*diff.get(2, 0), 5.0 - 4.0, 1e-12)); // 1.0
    assert!(approx_eq(*diff.get(0, 1), 2.0 - 1.0, 1e-12)); // 1.0

    assert!(approx_eq(*prod.get(0, 0), 1.0 * 10.0, 1e-12)); // 10
    assert!(approx_eq(*prod.get(2, 1), 6.0 * 5.0, 1e-12));  // 30

    assert!(approx_eq(*quot.get(1, 0), 3.0 / 2.0, 1e-12));
    assert!(approx_eq(*quot.get(2, 1), 6.0 / 5.0, 1e-12));

    // Shape check still consistent
    assert_eq!(sum.num_vec(), 3);
    assert_eq!(sum.dim(), 2);
}

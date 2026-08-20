use physics_in_parallel::prelude::advanced::{
    AntiSymmetricMatrix, DiagonalMatrix, LowerTriangularMatrix, StrictLowerTriangularMatrix,
    StrictUpperTriangularMatrix, SymmetricMatrix, UpperTriangularMatrix,
};

use crate::matrix_support::{assert_panics, print_matrix};

#[test]
fn diagonal_matrix_stores_only_diagonal_values_and_rejects_invalid_shape() {
    let mut diagonal = DiagonalMatrix::<i64>::empty(3, 3);
    diagonal.set(0, 0, 4);
    diagonal.set(1, 1, -2);
    diagonal.set(2, 1, 0);

    assert_eq!(diagonal.shape(), [3, 3]);
    assert_eq!(diagonal.get(0, 0), 4);
    assert_eq!(diagonal.get(1, 1), -2);
    assert_eq!(diagonal.get(0, 1), 0);
    assert_eq!(diagonal.trace(), 2);
    print_matrix("diagonal matrix", &diagonal);

    assert_panics(|| {
        let _ = DiagonalMatrix::<i64>::empty(2, 3);
    });
    assert_panics(move || {
        let mut invalid = diagonal;
        invalid.set(0, 2, 7);
    });
}

#[test]
fn symmetric_matrix_mirrors_canonical_entries_and_preserves_symmetric_ops() {
    let mut symmetric = SymmetricMatrix::<i64>::empty(4, 4);
    symmetric.set(0, 3, 8);
    symmetric.set(2, 1, -5);

    assert_eq!(symmetric.get(0, 3), 8);
    assert_eq!(symmetric.get(3, 0), 8);
    assert_eq!(symmetric.get(2, 1), -5);
    assert_eq!(symmetric.get(1, 2), -5);
    print_matrix("symmetric matrix", &symmetric);

    let doubled = symmetric.add(&symmetric);
    assert_eq!(doubled.get(0, 3), 16);
    assert_eq!(doubled.get(3, 0), 16);

    let transposed = symmetric.transpose();
    assert_eq!(transposed.get(0, 3), 8);
    assert_eq!(transposed.get(3, 0), 8);
}

#[test]
fn antisymmetric_matrix_infers_sign_and_preserves_sign_sensitive_ops() {
    let mut anti = AntiSymmetricMatrix::<i64>::empty(3, 3);
    anti.set(0, 2, 6);
    anti.set(2, 1, -4);

    assert_eq!(anti.get(0, 2), 6);
    assert_eq!(anti.get(2, 0), -6);
    assert_eq!(anti.get(2, 1), -4);
    assert_eq!(anti.get(1, 2), 4);
    assert_eq!(anti.get(1, 1), 0);
    print_matrix("antisymmetric matrix", &anti);

    let scaled = anti.scalar_mul(2);
    assert_eq!(scaled.get(0, 2), 12);
    assert_eq!(scaled.get(2, 0), -12);

    let transposed = anti.transpose();
    assert_eq!(transposed.get(0, 2), -6);
    assert_eq!(transposed.get(2, 0), 6);

    assert_panics(move || {
        let mut invalid = anti;
        invalid.set(1, 1, 3);
    });
}

#[test]
fn triangular_matrix_variants_respect_orientation_and_diagonal_policy() {
    let mut upper = UpperTriangularMatrix::<i64>::empty(3, 3);
    upper.set(0, 2, 9);
    upper.set(1, 1, 5);
    upper.set(2, 0, 0);
    assert_eq!(upper.get(0, 2), 9);
    assert_eq!(upper.get(1, 1), 5);
    assert_eq!(upper.get(2, 0), 0);
    print_matrix("upper triangular matrix", &upper);

    let mut strict_upper = StrictUpperTriangularMatrix::<i64>::empty(3, 3);
    strict_upper.set(0, 1, 7);
    strict_upper.set(1, 1, 0);
    assert_eq!(strict_upper.get(0, 1), 7);
    assert_eq!(strict_upper.get(1, 1), 0);
    print_matrix("strict upper triangular matrix", &strict_upper);

    let mut lower = LowerTriangularMatrix::<i64>::empty(3, 3);
    lower.set(2, 0, 11);
    lower.set(2, 2, 1);
    assert_eq!(lower.get(2, 0), 11);
    assert_eq!(lower.get(0, 2), 0);
    assert_eq!(lower.get(2, 2), 1);
    print_matrix("lower triangular matrix", &lower);

    let mut strict_lower = StrictLowerTriangularMatrix::<i64>::empty(3, 3);
    strict_lower.set(2, 0, -3);
    strict_lower.set(1, 1, 0);
    assert_eq!(strict_lower.get(2, 0), -3);
    assert_eq!(strict_lower.get(1, 1), 0);
    print_matrix("strict lower triangular matrix", &strict_lower);

    assert_panics(move || {
        let mut invalid = strict_upper;
        invalid.set(1, 1, 3);
    });
    assert_panics(move || {
        let mut invalid = lower;
        invalid.set(0, 2, 5);
    });
}

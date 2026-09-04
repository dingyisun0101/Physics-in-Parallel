/*!
Structured matrix backends.

Structured backends store only canonical entries and derive the remaining
logical matrix values from mathematical rules. This saves space and, more
importantly, prevents states that violate the declared structure.
*/

use core::ops::Neg;

use crate::math::scalar::Scalar;
use crate::math::tensor::rank_n::{Sparse, Tensor};

use super::generic::Matrix;
use super::matrix_backend_trait::{MatrixBackend, wrap_axis_index};

/// Diagonal matrix backend. Only diagonal entries are stored.
#[derive(Debug, Clone)]
pub struct Diagonal<T: Scalar> {
    size: usize,
    diagonal: Vec<T>,
}

/// Symmetric matrix backend with `a[i, j] = a[j, i]`.
#[derive(Debug, Clone)]
pub struct Symmetric<T: Scalar> {
    size: usize,
    canonical: Tensor<T, Sparse>,
}

/// Antisymmetric matrix backend with `a[i, j] = -a[j, i]` and zero diagonal.
#[derive(Debug, Clone)]
pub struct AntiSymmetric<T: Scalar> {
    size: usize,
    canonical: Tensor<T, Sparse>,
}

/// Triangular matrix backend.
///
/// `UPPER = true` stores entries above the diagonal. `UPPER = false` stores
/// entries below the diagonal. `INCLUDE_DIAGONAL` controls whether diagonal
/// values are part of the stored support or forced to zero.
#[derive(Debug, Clone)]
pub struct Triangular<T: Scalar, const UPPER: bool, const INCLUDE_DIAGONAL: bool> {
    rows: usize,
    cols: usize,
    canonical: Tensor<T, Sparse>,
}

pub type DiagonalMatrix<T> = Matrix<T, Diagonal<T>>;
pub type SymmetricMatrix<T> = Matrix<T, Symmetric<T>>;
pub type AntiSymmetricMatrix<T> = Matrix<T, AntiSymmetric<T>>;
pub type UpperTriangularMatrix<T> = Matrix<T, Triangular<T, true, true>>;
pub type StrictUpperTriangularMatrix<T> = Matrix<T, Triangular<T, true, false>>;
pub type LowerTriangularMatrix<T> = Matrix<T, Triangular<T, false, true>>;
pub type StrictLowerTriangularMatrix<T> = Matrix<T, Triangular<T, false, false>>;

impl<T: Scalar> Diagonal<T> {
    #[inline]
    fn wrap(&self, index: isize) -> usize {
        wrap_axis_index(index, self.size)
    }
}

impl<T: Scalar> MatrixBackend<T> for Diagonal<T> {
    fn diagonal_data(&self) -> Option<&[T]> {
        Some(&self.diagonal)
    }

    #[inline]
    fn empty(rows: usize, cols: usize) -> Self {
        assert_eq!(
            rows, cols,
            "diagonal matrix must be square; got {rows}x{cols}"
        );
        assert!(rows > 0, "diagonal matrix size must be nonzero");
        Self {
            size: rows,
            diagonal: vec![T::zero(); rows],
        }
    }

    #[inline]
    fn rows(&self) -> usize {
        self.size
    }

    #[inline]
    fn cols(&self) -> usize {
        self.size
    }

    #[inline]
    fn get(&self, row: isize, col: isize) -> T {
        let row = self.wrap(row);
        let col = self.wrap(col);
        if row == col {
            self.diagonal[row]
        } else {
            T::zero()
        }
    }

    #[inline]
    fn set(&mut self, row: isize, col: isize, value: T) {
        let row = self.wrap(row);
        let col = self.wrap(col);
        if row == col {
            self.diagonal[row] = value;
        } else {
            assert!(
                value == T::zero(),
                "cannot store nonzero off-diagonal value in diagonal matrix"
            );
        }
    }

    #[inline]
    fn fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.diagonal.fill(value);
    }
}

impl<T: Scalar> MatrixBackend<T> for Symmetric<T> {
    #[inline]
    fn empty(rows: usize, cols: usize) -> Self {
        assert_eq!(
            rows, cols,
            "symmetric matrix must be square; got {rows}x{cols}"
        );
        Self {
            size: rows,
            canonical: Tensor::<T, Sparse>::empty(&[rows, cols]),
        }
    }

    #[inline]
    fn rows(&self) -> usize {
        self.size
    }

    #[inline]
    fn cols(&self) -> usize {
        self.size
    }

    #[inline]
    fn get(&self, row: isize, col: isize) -> T {
        let row = wrap_axis_index(row, self.size);
        let col = wrap_axis_index(col, self.size);
        let (row, col) = canonical_pair(row, col);
        self.canonical.get(&[row as isize, col as isize])
    }

    #[inline]
    fn set(&mut self, row: isize, col: isize, value: T) {
        let row = wrap_axis_index(row, self.size);
        let col = wrap_axis_index(col, self.size);
        let (row, col) = canonical_pair(row, col);
        self.canonical.set(&[row as isize, col as isize], value);
    }
}

impl<T> MatrixBackend<T> for AntiSymmetric<T>
where
    T: Scalar + Neg<Output = T>,
{
    #[inline]
    fn empty(rows: usize, cols: usize) -> Self {
        assert_eq!(
            rows, cols,
            "antisymmetric matrix must be square; got {rows}x{cols}"
        );
        Self {
            size: rows,
            canonical: Tensor::<T, Sparse>::empty(&[rows, cols]),
        }
    }

    #[inline]
    fn rows(&self) -> usize {
        self.size
    }

    #[inline]
    fn cols(&self) -> usize {
        self.size
    }

    #[inline]
    fn get(&self, row: isize, col: isize) -> T {
        let row = wrap_axis_index(row, self.size);
        let col = wrap_axis_index(col, self.size);
        if row == col {
            return T::zero();
        }
        if row < col {
            self.canonical.get(&[row as isize, col as isize])
        } else {
            -self.canonical.get(&[col as isize, row as isize])
        }
    }

    #[inline]
    fn set(&mut self, row: isize, col: isize, value: T) {
        let row = wrap_axis_index(row, self.size);
        let col = wrap_axis_index(col, self.size);
        if row == col {
            assert!(
                value == T::zero(),
                "antisymmetric matrix diagonal is always zero"
            );
            return;
        }
        if row < col {
            self.canonical.set(&[row as isize, col as isize], value);
        } else {
            self.canonical.set(&[col as isize, row as isize], -value);
        }
    }
}

impl<T: Scalar, const UPPER: bool, const INCLUDE_DIAGONAL: bool> MatrixBackend<T>
    for Triangular<T, UPPER, INCLUDE_DIAGONAL>
{
    #[inline]
    fn empty(rows: usize, cols: usize) -> Self {
        assert!(
            rows > 0 && cols > 0,
            "triangular matrix shape must be nonzero"
        );
        Self {
            rows,
            cols,
            canonical: Tensor::<T, Sparse>::empty(&[rows, cols]),
        }
    }

    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    fn get(&self, row: isize, col: isize) -> T {
        let row = wrap_axis_index(row, self.rows);
        let col = wrap_axis_index(col, self.cols);
        if triangular_contains::<UPPER, INCLUDE_DIAGONAL>(row, col) {
            self.canonical.get(&[row as isize, col as isize])
        } else {
            T::zero()
        }
    }

    #[inline]
    fn set(&mut self, row: isize, col: isize, value: T) {
        let row = wrap_axis_index(row, self.rows);
        let col = wrap_axis_index(col, self.cols);
        if triangular_contains::<UPPER, INCLUDE_DIAGONAL>(row, col) {
            self.canonical.set(&[row as isize, col as isize], value);
        } else {
            assert!(
                value == T::zero(),
                "cannot store nonzero value outside triangular support"
            );
        }
    }

    fn fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        if value == T::zero() {
            self.canonical.fill(T::zero());
            return;
        }
        for row in 0..self.rows {
            for col in 0..self.cols {
                if triangular_contains::<UPPER, INCLUDE_DIAGONAL>(row, col) {
                    self.canonical.set(&[row as isize, col as isize], value);
                }
            }
        }
    }
}

#[inline]
fn canonical_pair(row: usize, col: usize) -> (usize, usize) {
    if row <= col { (row, col) } else { (col, row) }
}

#[inline]
fn triangular_contains<const UPPER: bool, const INCLUDE_DIAGONAL: bool>(
    row: usize,
    col: usize,
) -> bool {
    if UPPER {
        col > row || (INCLUDE_DIAGONAL && col == row)
    } else {
        row > col || (INCLUDE_DIAGONAL && row == col)
    }
}

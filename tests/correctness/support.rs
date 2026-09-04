use std::collections::BTreeMap;

/// Every oracle consumes plain inputs, never another PiP operation's result.
#[derive(Default)]
pub struct Report {
    rows: BTreeMap<String, (usize, f64, f64)>,
}
impl Report {
    pub fn close(&mut self, label: &str, actual: &[f64], expected: &[f64], atol: f64, rtol: f64) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        // Aggregate lengths while retaining the full case label in failures.
        let summary = label.split(" n=").next().unwrap();
        let row = self.rows.entry(summary.to_owned()).or_default();
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            row.0 += 1;
            if !a.is_finite() || !e.is_finite() {
                assert!(
                    a == e || a.is_nan() && e.is_nan(),
                    "{label}[{index}]: actual={a:?}, expected={e:?}"
                );
                continue;
            }
            let error = (a - e).abs();
            let bound = atol + rtol * e.abs();
            row.1 = row.1.max(error);
            row.2 = row.2.max(if bound == 0.0 { 0.0 } else { error / bound });
            assert!(
                error <= bound,
                "{label}[{index}]: actual={a:.17e}, expected={e:.17e}, abs_error={error:.4e}, allowed={bound:.4e} (atol={atol:e}, rtol={rtol:e})"
            );
        }
    }
    pub fn exact<T: std::fmt::Debug + PartialEq>(&mut self, label: &str, actual: T, expected: T) {
        assert_eq!(actual, expected, "{label}");
        self.rows.entry(label.to_owned()).or_default().0 += 1;
    }
    pub fn finish(self) {
        for (label, (count, max_abs, max_fraction)) in self.rows {
            println!(
                "REFERENCE PASS {label} checks={count} max_abs_error={max_abs:.6e} max_tolerance_fraction={max_fraction:.6e}"
            );
        }
    }
}

pub const BACKENDS: [physics_in_parallel::math::Backend; 2] = [
    physics_in_parallel::math::Backend::Dense,
    physics_in_parallel::math::Backend::Sparse,
];
pub const ATOL: f64 = 2e-12;
pub const RTOL: f64 = 2e-11;

pub fn data(n: usize, salt: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if (i + salt).is_multiple_of(5) {
                0.0
            } else {
                (((i * 73 + salt * 19) % 509) as f64 - 254.0) / 37.0
            }
        })
        .collect()
}

pub fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                out[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    out
}

pub fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    (0..cols)
        .flat_map(|j| (0..rows).map(move |i| a[i * cols + j]))
        .collect()
}

pub fn coords(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    result
}
pub fn flat(coords: &[usize], shape: &[usize]) -> usize {
    coords
        .iter()
        .zip(shape)
        .fold(0, |n, (&x, &width)| n * width + x)
}

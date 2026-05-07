use std::panic;
use std::sync::Mutex;

use physics_in_parallel::math::scalar::Scalar;
use physics_in_parallel::math::tensor::{Matrix, MatrixBackend};

static PANIC_HOOK_LOCK: Mutex<()> = Mutex::new(());

pub fn print_matrix<T, B>(label: &str, matrix: &Matrix<T, B>)
where
    T: Scalar + Copy,
    B: MatrixBackend<T>,
{
    println!("\n{label}");
    matrix.print();
}

pub fn assert_panics<F>(f: F)
where
    F: FnOnce() + panic::UnwindSafe,
{
    let _guard = PANIC_HOOK_LOCK
        .lock()
        .expect("panic-hook suppression lock should not be poisoned");
    let original_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(f);
    panic::set_hook(original_hook);
    assert!(result.is_err(), "operation should have panicked");
}

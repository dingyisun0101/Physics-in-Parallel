//! Allocation regression for the reusable dense public operations.
use physics_in_parallel::math::{Backend, Matrix, Tensor};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}
fn count() {
    let _ = ACTIVE.try_with(|active| {
        if active.get() {
            let _ = COUNT.try_with(|count| count.set(count.get() + 1));
        }
    });
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: forwarding the allocator's original layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count();
        // SAFETY: forwarding the original valid allocation and requested size.
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: forwarding the matching allocator and layout.
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn warmed_dense_outputs_do_not_allocate() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let a = Tensor::filled(&[100_003], Backend::Dense, 2.0_f64).unwrap();
        let b = a.clone();
        let mut out = a.clone();
        let matrix = Matrix::filled(5, 7, Backend::Dense, 2.0_f64).unwrap();
        let rhs = Matrix::filled(7, 3, Backend::Dense, 3.0_f64).unwrap();
        let mut product = Matrix::zeros(5, 3, Backend::Dense).unwrap();
        a.add_into(&b, &mut out).unwrap();
        matrix.matmul_into(&rhs, &mut product).unwrap();
        COUNT.set(0);
        ACTIVE.set(true);
        out.fill(1.0);
        out.map_in_place(|x| x + 1.0);
        a.add_into(&b, &mut out).unwrap();
        matrix.matmul_into(&rhs, &mut product).unwrap();
        ACTIVE.set(false);
        assert_eq!(COUNT.get(), 0);
        assert_eq!(product.get(0, 0).unwrap(), 42.0);
    });
}

use super::support::*;
use physics_in_parallel::prelude::basic::*;

macro_rules! real_suite {
    ($name:ident, $ty:ty, $eps:expr) => {
        #[test]
        fn $name() {
            let mut report = Report::default();
            let tol = 32.0 * $eps;
            for n in [1, 3, 31, 32, 127, 128, 257, 16403] {
                let av: Vec<$ty> = data(n, 3).into_iter().map(|v| v as $ty).collect();
                let bv: Vec<$ty> = data(n, 7).into_iter().map(|v| (v.abs() + 0.25) as $ty).collect();
                let a64: Vec<_> = av.iter().map(|&v| v as f64).collect();
                let b64: Vec<_> = bv.iter().map(|&v| v as f64).collect();
                for left in BACKENDS {
                    for right in BACKENDS {
                        for destination in BACKENDS {
                            macro_rules! container {
                                ($kind:ident, $make:expr) => {{
                                    let make = $make;
                                    let a = make(left, av.clone());
                                    let b = make(right, bv.clone());
                                    let mut out = make(destination, vec![99.0; n]);
                                    macro_rules! operation {
                                        ($alloc:ident, $into:ident, $operator:tt) => {{
                                            let expected: Vec<_> = a64.iter().zip(&b64).map(|(&a, &b)| a $operator b).collect();
                                            let allocated = a.$alloc(&b).unwrap();
                                            let label = format!("{}.{}.{} {:?}/{:?}/{:?} n={n}", stringify!($ty), stringify!($kind), stringify!($alloc), left, right, destination);
                                            report.close(&label, &allocated.values().map(|v| v as f64).collect::<Vec<_>>(), &expected, tol, tol);
                                            a.$into(&b, &mut out).unwrap();
                                            report.close(&label, &out.values().map(|v| v as f64).collect::<Vec<_>>(), &expected, tol, tol);
                                            assert_eq!(out.backend(), destination);
                                            assert_eq!(allocated.backend(), left);
                                        }};
                                    }
                                    operation!(add, add_into, +);
                                    operation!(subtract, subtract_into, -);
                                    operation!(multiply, multiply_into, *);
                                    operation!(divide, divide_into, /);
                                    let expected: Vec<_> = a64.iter().map(|v| v * -1.75).collect();
                                    report.close(concat!(stringify!($ty), ".", stringify!($kind), ".scale"), &a.scale(-1.75).values().map(|v| v as f64).collect::<Vec<_>>(), &expected, tol, tol);
                                    report.close("input remains unchanged", &a.values().map(|v| v as f64).collect::<Vec<_>>(), &a64, 0.0, 0.0);
                                }};
                            }
                            container!(Tensor, |backend, values| Tensor::from_values(&[n], backend, values).unwrap());
                            container!(Matrix, |backend, values| Matrix::from_values(1, n, backend, values).unwrap());
                            container!(VectorList, |backend, values| VectorList::from_values(1, n, backend, values).unwrap());
                        }
                    }
                    let a = Tensor::from_values(&[n], left, av.clone()).unwrap();
                    let b = Tensor::from_values(&[n], left, bv.clone()).unwrap();
                    let sum: f64 = a64.iter().sum();
                    let dot: f64 = a64.iter().zip(&b64).map(|(a, b)| a * b).sum();
                    let squares: f64 = a64.iter().map(|v| v * v).sum();
                    // Reduction tolerance scales with the L1 magnitude, including cancellation.
                    let sum_abs = a64.iter().map(|v| v.abs()).sum::<f64>();
                    let dot_abs = a64.iter().zip(&b64).map(|(a,b)| (a*b).abs()).sum::<f64>();
                    report.close(concat!(stringify!($ty), ".Tensor.sum"), &[a.sum() as f64], &[sum], tol * sum_abs, tol);
                    report.close(concat!(stringify!($ty), ".Tensor.dot/hermitian_dot"), &[a.dot(&b).unwrap() as f64, a.hermitian_dot(&b).unwrap() as f64], &[dot, dot], tol * dot_abs, tol);
                    report.close(concat!(stringify!($ty), ".Tensor.norm"), &[a.norm_squared_real() as f64, a.norm() as f64], &[squares, squares.sqrt()], tol * squares, tol);
                    for (label, actual, expected) in [
                        ("abs", a.abs(), a64.iter().map(|v| v.abs()).collect::<Vec<_>>()),
                        ("norm_squared", a.norm_squared(), a64.iter().map(|v| v*v).collect()),
                        ("sqrt", b.sqrt(), b64.iter().map(|v| v.sqrt()).collect()),
                        ("conjugate", a.conjugate(), a64.clone()),
                        ("map", a.map(|v| v * v + 0.25), a64.iter().map(|v| v*v + 0.25).collect()),
                    ] {
                        report.close(&format!("{}.Tensor.{label}", stringify!($ty)), &actual.values().map(|v| v as f64).collect::<Vec<_>>(), &expected, tol, tol);
                    }
                    let mut mapped = a.clone();
                    mapped.map_in_place(|v| v * v + 0.25);
                    report.close("Tensor.map_in_place", &mapped.values().map(|v| v as f64).collect::<Vec<_>>(), &a64.iter().map(|v| v*v+0.25).collect::<Vec<_>>(), tol, tol);
                    let cast = a.cast::<f64>().unwrap();
                    report.close("Tensor.cast", &cast.values().collect::<Vec<_>>(), &a64, 0.0, 0.0);
                }
            }
            report.finish();
        }
    };
}
real_suite!(f32_arithmetic_and_reductions, f32, f32::EPSILON as f64);
real_suite!(f64_arithmetic_and_reductions, f64, f64::EPSILON);

#[test]
fn construction_coordinates_builders_storage_and_serde() {
    let mut report = Report::default();
    for shape in [vec![], vec![1], vec![3, 5], vec![2, 3, 7]] {
        let n = shape.iter().product();
        let values = data(n, 3);
        for backend in BACKENDS {
            let mut builder = Tensor::empty(&shape, backend).unwrap();
            for (i, &v) in values.iter().enumerate() { builder.set(&coords(i, &shape), v).unwrap(); }
            let built = builder.finish().unwrap();
            let generated = Tensor::from_fn(&shape, backend, |c| values[flat(c, &shape)]).unwrap();
            let entries = Tensor::from_entries(&shape, backend, values.iter().enumerate().map(|(i,&v)| (coords(i,&shape),v))).unwrap();
            for t in [built, generated, entries] {
                report.close("Tensor constructors", &t.values().collect::<Vec<_>>(), &values, 0.0, 0.0);
                assert_eq!(t.rank(), shape.len()); assert_eq!(t.size(), n); assert_eq!(t.shape(), shape);
                for (i,&v) in values.iter().enumerate() { assert_eq!(t.get(&coords(i,&shape)).unwrap(), v); }
                let decoded: Tensor<f64> = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
                report.close("Tensor serde", &decoded.values().collect::<Vec<_>>(), &values, 0.0, 0.0);
                let mut converted = t.clone();
                for target in BACKENDS { converted.set_backend(target); report.close("Tensor backend conversion", &converted.values().collect::<Vec<_>>(), &values, 0.0, 0.0); }
                converted.fill(2.5);
                report.close("Tensor.fill", &converted.values().collect::<Vec<_>>(), &vec![2.5;n], 0.0,0.0);
            }
        }
    }
    for backend in BACKENDS {
        let mut mb = Matrix::empty(3, 5, backend).unwrap();
        let mut vb = VectorList::empty(5, 3, backend).unwrap();
        let values = data(15, 2);
        for row in 0..3 { vb.set_vector(row, &values[row*5..row*5+5]).unwrap(); for col in 0..5 { mb.set(row,col,values[row*5+col]).unwrap(); } }
        let matrix = mb.finish().unwrap(); let vectors = vb.finish().unwrap();
        report.close("Matrix.builder", &matrix.values().collect::<Vec<_>>(), &values, 0.0,0.0);
        report.close("VectorList.builder", &vectors.values().collect::<Vec<_>>(), &values, 0.0,0.0);
        let m: Matrix<f64> = serde_json::from_str(&serde_json::to_string(&matrix).unwrap()).unwrap();
        let v: VectorList<f64> = serde_json::from_str(&serde_json::to_string(&vectors).unwrap()).unwrap();
        report.close("Matrix/VectorList serde", &m.values().chain(v.values()).collect::<Vec<_>>(), &values.iter().chain(&values).copied().collect::<Vec<_>>(), 0.0,0.0);
        let entries = (0..15).map(|i| (i/5,i%5,values[i]));
        let m = Matrix::from_entries(3,5,backend,entries.clone()).unwrap();
        let v = VectorList::from_entries(5,3,backend,entries).unwrap();
        report.close("Matrix/VectorList.from_entries", &m.values().chain(v.values()).collect::<Vec<_>>(), &values.iter().chain(&values).copied().collect::<Vec<_>>(),0.0,0.0);
    }
    report.finish();
}

#[test]
fn matrix_tensor_products_transposes_and_vector_batches() {
    let mut report = Report::default();
    for (m,k,n) in [(1,1,1),(3,7,5),(17,33,9),(33,65,35),(65,65,65)] {
        let av = data(m*k,2); let bv=data(k*n,7);
        let expected=matmul(&av,&bv,m,k,n);
        for left in BACKENDS { for right in BACKENDS {
            let a=Matrix::from_values(m,k,left,av.clone()).unwrap();
            let b=Matrix::from_values(k,n,right,bv.clone()).unwrap();
            let ta=Tensor::from_values(&[m,k],left,av.clone()).unwrap();
            let tb=Tensor::from_values(&[k,n],right,bv.clone()).unwrap();
            report.close("Matrix.matmul", &a.matmul(&b).unwrap().values().collect::<Vec<_>>(),&expected,ATOL,RTOL);
            report.close("Tensor.matmul", &ta.matmul(&tb).unwrap().values().collect::<Vec<_>>(),&expected,ATOL,RTOL);
            let af=a.cast::<f32>().unwrap(); let bf=b.cast::<f32>().unwrap();
            let expected32=matmul(&av.iter().map(|&v|v as f32 as f64).collect::<Vec<_>>(), &bv.iter().map(|&v|v as f32 as f64).collect::<Vec<_>>(),m,k,n);
            report.close("f32.Matrix.matmul", &af.matmul(&bf).unwrap().values().map(f64::from).collect::<Vec<_>>(),&expected32,2e-4,2e-5);
            for destination in BACKENDS {
                let mut out=Matrix::filled(m,n,destination,99.0).unwrap(); a.matmul_into(&b,&mut out).unwrap();
                report.close("Matrix.matmul_into",&out.values().collect::<Vec<_>>(),&expected,ATOL,RTOL);
                let mut out=Tensor::filled(&[m,n],destination,99.0).unwrap(); ta.matmul_into(&tb,&mut out).unwrap();
                report.close("Tensor.matmul_into",&out.values().collect::<Vec<_>>(),&expected,ATOL,RTOL);
            }
            let tr=transpose(&av,m,k);
            report.close("Matrix.transpose/hermitian",&a.transpose().unwrap().values().chain(a.hermitian_transpose().unwrap().values()).collect::<Vec<_>>(),&tr.iter().chain(&tr).copied().collect::<Vec<_>>(),ATOL,RTOL);
            report.close("Tensor.transpose/hermitian",&ta.transpose().unwrap().values().chain(ta.hermitian_transpose().unwrap().values()).collect::<Vec<_>>(),&tr.iter().chain(&tr).copied().collect::<Vec<_>>(),ATOL,RTOL);
            let x=data(k*3,11); let mut output=vec![99.0;m*3];
            let batch_expected:Vec<_>=x.chunks(k).flat_map(|v|matmul(&av,v,m,k,1)).collect();
            a.mul_vectors_into(&x,&mut output).unwrap();
            report.close("Matrix.mul_vectors_into",&output,&batch_expected,ATOL,RTOL);
            a.mul_vector_into(&x[..k],&mut output[..m]).unwrap();
            report.close("Matrix.mul_vector_into",&output[..m],&batch_expected[..m],ATOL,RTOL);
            report.close("Matrix.trace/max_abs/abs", &[a.trace(), a.max_abs_real()], &[ (0..m.min(k)).map(|i|av[i*k+i]).sum(), av.iter().map(|v|v.abs()).fold(0.0,f64::max)],ATOL,RTOL);
            report.close("Matrix.abs",&a.abs().values().collect::<Vec<_>>(),&av.iter().map(|v|v.abs()).collect::<Vec<_>>(),ATOL,RTOL);
        }}
    }
    for backend in BACKENDS {
        let expected:Vec<_>=(0..49).map(|i|if i/7==i%7{1.0}else{0.0}).collect();
        report.close("Matrix/Tensor.identity",&Matrix::identity(7,backend).unwrap().values().chain(Tensor::identity(7,backend).unwrap().values()).collect::<Vec<f64>>(),&expected.iter().chain(&expected).copied().collect::<Vec<_>>(),0.0,0.0);
    }
    report.finish();
}

#[test]
fn vector_geometry_and_mutation() {
    let mut report=Report::default();
    for backend in BACKENDS { for dim in [1,2,3,7,17] { for n in [1,33,1031] {
        let av=data(n*dim,3);
        let a=VectorList::from_values(dim,n,backend,av.clone()).unwrap();
        let norms:Vec<_>=av.chunks(dim).map(|row|row.iter().map(|v|v*v).sum::<f64>().sqrt()).collect();
        report.close("VectorList.norms/norms_real",&a.norms().into_iter().chain(a.norms_real()).collect::<Vec<_>>(),&norms.iter().chain(&norms).copied().collect::<Vec<_>>(),ATOL,RTOL);
        let expected:Vec<_>=av.chunks(dim).zip(&norms).flat_map(|(row,&norm)|row.iter().map(move |&v|if norm==0.0{v}else{v/norm})).collect();
        let mut normalized=a.clone(); normalized.normalize().unwrap();
        report.close("VectorList.normalize",&normalized.values().collect::<Vec<_>>(),&expected,ATOL,RTOL);
        let (r,direction)=a.to_polar().unwrap();
        report.close("VectorList.to_polar radii",&r,&norms,ATOL,RTOL);
        report.close("VectorList.to_polar directions",&direction.values().collect::<Vec<_>>(),&expected,ATOL,RTOL);
        for i in [0,n-1] { let mut out=vec![99.0;dim]; a.vector_into(i,&mut out).unwrap(); report.close("VectorList.vector/vector_into",&a.vector(i).unwrap().into_iter().chain(out).collect::<Vec<_>>(),&av[i*dim..i*dim+dim].iter().cycle().take(2*dim).copied().collect::<Vec<_>>(),0.0,0.0); }
        let scales=data(n,13); let mut scaled=a.clone(); scaled.scale_vectors(&scales).unwrap();
        report.close("VectorList.scale_vectors",&scaled.values().collect::<Vec<_>>(),&av.iter().enumerate().map(|(i,v)|v*scales[i/dim]).collect::<Vec<_>>(),ATOL,RTOL);
        let mut changed=a.clone(); let mut reference=av.clone();
        for axis in 0..dim {
            let expected:Vec<_>=(0..n).map(|i|av[i*dim+axis]).collect(); let mut out=vec![99.0;n]; a.axis_into(axis,&mut out).unwrap();
            report.close("VectorList.axis/axis_into",&a.axis(axis).unwrap().into_iter().chain(out).collect::<Vec<_>>(),&expected.iter().chain(&expected).copied().collect::<Vec<_>>(),0.0,0.0);
            changed.set_axis(axis,&scales).unwrap(); for i in 0..n {reference[i*dim+axis]=scales[i];}
        }
        changed.set_vector(0,&vec![3.5;dim]).unwrap(); reference[..dim].fill(3.5);
        changed.set(n-1,dim-1,-7.0).unwrap(); reference[n*dim-1]=-7.0;
        report.close("VectorList.set_axis/set_vector/set",&changed.values().collect::<Vec<_>>(),&reference,0.0,0.0);
        let converted=changed.cast::<f32>().unwrap(); report.close("VectorList.cast",&converted.values().map(f64::from).collect::<Vec<_>>(),&reference,1e-6,1e-6);
    }}}
    for left in BACKENDS { for right in BACKENDS {
        let a=[1.3,-2.7,0.25]; let b=[-0.5,3.25,1.75];
        let ta=Tensor::from_values(&[3],left,a.to_vec()).unwrap(); let tb=Tensor::from_values(&[3],right,b.to_vec()).unwrap();
        let cross=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
        report.close("Tensor.cross",&ta.cross(&tb).unwrap().values().collect::<Vec<_>>(),&cross,ATOL,RTOL);
        let wedge:Vec<_>=(0..3).flat_map(|i|(0..3).map(move |j|a[i]*b[j]-a[j]*b[i])).collect();
        report.close("Tensor.wedge",&ta.wedge(&tb).unwrap().values().collect::<Vec<_>>(),&wedge,ATOL,RTOL);
    }}
    report.finish();
}

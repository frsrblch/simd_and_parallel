use criterion::*;
use simd_and_parallel::*;

const N: usize = 5_000;

fn scalar_magnitude_squared(c: &mut Criterion) {
    let x = (0..).into_iter()
        .map(|i| i as f64 * 0.9)
        .take(N)
        .collect::<Vec<_>>();

    let y = (0..).into_iter()
        .map(|i| i as f64 * 1.1)
        .take(N)
        .collect::<Vec<_>>();

    let mut m = (0..).into_iter().map(|_| 0f64).take(N).collect::<Vec<_>>();

    c.bench_function("scalar" , |b| b.iter(|| magnitude_squared_fallback(&x, &y, &mut m)));
}

fn parallel_scalar_magnitude_squared(c: &mut Criterion) {
    let x = (0..).into_iter()
        .map(|i| i as f64 * 0.9)
        .take(N)
        .collect::<Vec<_>>();

    let y = (0..).into_iter()
        .map(|i| i as f64 * 1.1)
        .take(N)
        .collect::<Vec<_>>();

    let mut m = (0..).into_iter().map(|_| 0f64).take(N).collect::<Vec<_>>();

    c.bench_function("par scalar" , |b| b.iter(|| parallel_magnitude_squared_fallback(&x, &y, &mut m)));
}

fn vector_magnitude_squared(c: &mut Criterion) {
    let x = (0..).into_iter()
        .map(|i| i as f64 * 0.9)
        .take(N)
        .fold(Vec::with_capacity(N), |mut v, i| {
            v.push(i);
            v
        });

    let y = (0..).into_iter()
        .map(|i| i as f64 * 1.1)
        .take(N)
        .fold(Vec::with_capacity(N), |mut v, i| {
            v.push(i);
            v
        });

    let mut m = (0..).into_iter().map(|_| 0f64).take(N).collect::<Vec<_>>();

    c.bench_function("vector" , |b| b.iter(|| magnitude_squared(&x, &y, &mut m)));
}

fn parallel_vector_magnitude_squared(c: &mut Criterion) {
    let x = (0..).into_iter()
        .map(|i| i as f64 * 0.9)
        .take(N)
        .collect::<Vec<_>>();

    let y = (0..).into_iter()
        .map(|i| i as f64 * 1.1)
        .take(N)
        .collect::<Vec<_>>();

    let mut m = (0..).into_iter().map(|_| 0f64).take(N).collect::<Vec<_>>();

    c.bench_function("par vector" , |b| b.iter(|| parallel_magnitude_squared(&x, &y, &mut m)));
}

fn packed_vector(c: &mut Criterion) {
    let v = (0..).into_iter()
        .map(|i| f64x2::new(i as f64 * 0.9, i as f64 * 1.1))
        .take(N)
        .collect::<Vec<_>>();

    let mut m = (0..).into_iter().map(|_| 0f64).take(N).collect::<Vec<_>>();

    c.bench_function("packed vector" , |b| b.iter(|| packed_magnitude_squared(&v, &mut m)));
}

fn vector_type(c: &mut Criterion) {
    let v = (0..).into_iter()
        .map(|i| {
            let x = i as f64 * 0.9;
            let y = i as f64 * 1.1;
            Vector { x, y }
        })
        .take(N)
        .collect::<Vec<_>>();

    let mut m = (0..).into_iter().map(|_| 0f64).take(N).collect::<Vec<_>>();

    c.bench_function("vector type", |b| b.iter(|| Vector::msqrd(&v, &mut m)));
}

fn par_vector_type(c: &mut Criterion) {
    let v = (0..).into_iter()
        .map(|i| {
            let x = i as f64 * 0.9;
            let y = i as f64 * 1.1;
            Vector { x, y }
        })
        .take(N)
        .collect::<Vec<_>>();

    let mut m = (0..).into_iter().map(|_| 0f64).take(N).collect::<Vec<_>>();

    c.bench_function("par vector type", |b| b.iter(|| Vector::par_msqrd(&v, &mut m)));
}

criterion_group!(
    functions,
    scalar_magnitude_squared,
    vector_magnitude_squared,
    parallel_scalar_magnitude_squared,
    parallel_vector_magnitude_squared,
    packed_vector,
    vector_type,
    par_vector_type
);

criterion_main! {
    functions
}
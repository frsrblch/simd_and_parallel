use criterion::*;
use oorandom::Rand64;
use simd_and_parallel::*;
use rayon::prelude::*;

const ATTRACTORS: usize = 100;
const BODIES: usize = 4096*16;

criterion_main! {
    add_assign,
    gravity,
    movement,
    add_assign_mul,
}

criterion_group!(
    add_assign,
    vectors_add_assign,
    vec2_add_assign,
    vec3_add_assign,
    vectors3_add_assign,
);

fn get_floats_vec() -> Vec<(Float, Float)> {
    (0..).into_iter()
        .map(|i| (i as Float * 0.9, i as Float * 1.1))
        .take(BODIES)
        .collect()
}

fn vectors_add_assign(c: &mut Criterion) {
    let mut lhs = get_floats_vec()
        .into_iter()
        .map(|(x, y)| Vector { x, y })
        .fold(Vectors::default(), |mut v, val| {
            v.val.push(val);
            v
        });

    let rhs = lhs.clone();

    c.bench_function(
        "vectors add assign",
        |b| b.iter(|| {
            lhs += &rhs;
        })
    );
}

fn vec2_add_assign(c: &mut Criterion) {
    let mut lhs = get_floats_vec()
        .into_iter()
        .fold(Vec2::default(), |mut v, (x, y)| {
            v.x.push(x);
            v.y.push(y);
            v
        });

    let rhs = lhs.clone();

    c.bench_function(
        "vec2 add assign",
        |b| b.iter(|| {
            lhs += &rhs;
        })
    );
}

fn vec3_add_assign(c: &mut Criterion) {
    let rng = &mut Rand64::new(0);

    let mut lhs = get_vec3(rng, BODIES);
    let rhs = get_vec3(rng, BODIES);

    c.bench_function(
        "vec3 add assign",
        |b| b.iter(|| {
            lhs += &rhs;
        })
    );
}

fn vectors3_add_assign(c: &mut Criterion) {
    let rng = &mut Rand64::new(0);

    let mut lhs = get_vectors3(rng, BODIES);
    let rhs = get_vectors3(rng, BODIES);

    c.bench_function(
        "vectors3 add assign",
        |b| b.iter(|| {
            lhs += &rhs;
        })
    );
}

fn get_vec2(rng: &mut Rand64, n: usize) -> Vec2<Float> {
    (0..n)
        .into_iter()
        .fold(Vec2::default(), |mut v, _| {
            v.x.push(rng.rand_float() as Float);
            v.y.push(rng.rand_float() as Float);
            v
        })
}

fn get_vec2_simd(n: usize) -> Vec2<f32x8> {
    assert_eq!(4, std::mem::size_of::<Float>());
    assert_eq!(0, n % 8);

    let mut rng = thread_rng();

    (0..n/8)
        .into_iter()
        .fold(Vec2::default(), |mut v, _| {
            v.x.push(f32x8::new(rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen()));
            v.y.push(f32x8::new(rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen()));
            v
        })
}

fn get_vec(rng: &mut Rand64, n: usize) -> Vec1<Float> {
    Vec1 {
        val: (0..n)
            .into_iter()
            .map(|_| rng.rand_float() as Float)
            .collect()
    }
}

fn get_vec3(rng: &mut Rand64, n: usize) -> Vec3 {
    Vec3 {
        x: get_floats(rng, n),
        y: get_floats(rng, n),
        z: get_floats(rng, n),
    }
}

fn get_floats(rng: &mut Rand64, n: usize) -> Vec<Float> {
    std::iter::repeat_with(|| rng.rand_float() as Float)
        .take(n)
        .collect()
}

fn get_vectors3(rng: &mut Rand64, n: usize) -> Vectors3 {
    Vectors3 {
        val: std::iter::repeat_with(|| get_vector3(rng))
            .take(n)
            .collect()
    }
}

fn get_vector3(rng: &mut Rand64) -> Vector3 {
    Vector3 {
        x: rng.rand_float() as Float,
        y: rng.rand_float() as Float,
        z: rng.rand_float() as Float,
    }
}

fn get_link(rng: &mut Rand64, n: usize, range: usize) -> Vec<usize> {
    (0..n)
        .into_iter()
        .map(|_| rng.rand_range(0..range as u64) as usize)
        .collect()
}

fn get_empty_vec2(n: usize) -> Vec2<Float> {
    Vec2 {
        x: (0..n).into_iter().map(|_| 0.0).collect(),
        y: (0..n).into_iter().map(|_| 0.0).collect(),
    }
}

fn get_empty_vec2_simd(n: usize) -> Vec2<f32x8> {
    assert_eq!(4, std::mem::size_of::<Float>());
    assert_eq!(0, n % 8);

    Vec2 {
        x: (0..n/8).into_iter().map(|_| Default::default()).collect(),
        y: (0..n/8).into_iter().map(|_| Default::default()).collect(),
    }
}

fn get_empty_vec(n: usize) -> Vec1<Float> {
    Vec1 {
        val: (0..n).into_iter().map(|_| 0.0).collect()
    }
}

fn get_empty_vec_simd(n: usize) -> Vec<f32x8> {
    assert_eq!(4, std::mem::size_of::<Float>());
    assert_eq!(0, n % 8);

    (0..n).into_iter().map(|_| Default::default()).collect()
}

criterion_group!(
    gravity,
    vec2_gravity,
    vec2_gravity_simd,
    vec2_gravity_par,
    vec2_gravity_par_simd,
    vector_gravity,
    vector_gravity_par,
    vec2_gravity_no_buffer,
);

const G: Float = 6.674e-11;
const DT: Float = 0.1;

fn vec2_gravity(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let att_mass = get_vec(&mut rng, ATTRACTORS);
    let att_pos = get_vec2(&mut rng, ATTRACTORS);

    let body_pos = get_vec2(&mut rng, BODIES);
    let mut body_vel = get_vec2(&mut rng, BODIES);
    let body_att = get_link(&mut rng,  BODIES, ATTRACTORS);
    let mut parent_pos = get_empty_vec2(BODIES);
    let mut parent_mass = get_empty_vec(BODIES);

    c.bench_function(
        "vec2 gravity",
        |b| b.iter(|| {
            parent_pos.get_from(&body_att, &att_pos);
            parent_mass.get_from(&body_att, &att_mass);

            parent_pos -= &body_pos;

            body_vel.x.iter_mut()
                .zip(body_vel.y.iter_mut())
                .zip(parent_pos.x.iter().zip(parent_pos.y.iter()))
                .zip(parent_mass.val.iter())
                .for_each(|(((bvx, bvy), (px, py)), pm)| {
                    let dist_squared = (*px * *px) + (*py * *py);
                    let dist = dist_squared.sqrt();
                    let ux = *px / dist;
                    let uy = *py / dist;
                    let a = G * DT * *pm / dist_squared;
                    *bvx += ux * a;
                    *bvy += uy * a;
                })
        })
    );
}

use packed_simd::*;
use rand::{thread_rng, Rng};

fn vec2_gravity_simd(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let att_mass = get_vec(&mut rng, ATTRACTORS);
    let att_pos = get_vec2(&mut rng, ATTRACTORS);

    let body_pos = get_vec2_simd(BODIES);
    let mut body_vel = get_vec2_simd(BODIES);
    let body_att = get_link(&mut rng,  BODIES, ATTRACTORS);
    let mut parent_pos = get_empty_vec2_simd(BODIES);
    let mut parent_mass = get_empty_vec_simd(BODIES);

    c.bench_function(
        "vec2 gravity simd",
        |b| b.iter(|| {

            // parent_pos.get_from(&body_att, &att_pos);
            parent_pos.x.iter_mut()
                .zip(parent_pos.y.iter_mut())
                .zip(body_att.chunks_exact(8))
                .for_each(|((px, py), p)| {
                    p.iter()
                        .enumerate()
                        .for_each(|(i, parent)| {
                            if let (Some(p_pos_x), Some(p_pos_y)) = (att_pos.x.get(*parent), att_pos.y.get(*parent)) {
                                let _ = px.replace(i, *p_pos_x);
                                let _ = py.replace(i, *p_pos_y);
                            }
                        });
                });

            parent_mass.iter_mut()
                .zip(body_att.chunks_exact(8))
                .for_each(|(pm, p)| {
                    p.iter()
                        .enumerate()
                        .for_each(|(i, parent)| {
                            if let Some(p_mass) = att_mass.val.get(*parent) {
                                let _ = pm.replace(i, *p_mass);
                            }
                        });
                });

            parent_pos -= &body_pos;

            body_vel.x.iter_mut()
                .zip(body_vel.y.iter_mut())
                .zip(parent_pos.x.iter().zip(parent_pos.y.iter()))
                .zip(parent_mass.iter())
                .for_each(|(((bvx, bvy), (px, py)), pm)| {
                    let dist_squared = (*px * *px) + (*py * *py);
                    let dist = dist_squared.sqrt();
                    let ux = *px / dist;
                    let uy = *py / dist;
                    let a = *pm * G * DT / dist_squared;
                    *bvx += ux * a;
                    *bvy += uy * a;
                })
        })
    );
}

fn vec2_gravity_par_simd(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let att_mass = get_vec(&mut rng, ATTRACTORS);
    let att_pos = get_vec2(&mut rng, ATTRACTORS);

    let body_pos = get_vec2_simd(BODIES);
    let mut body_vel = get_vec2_simd(BODIES);
    let body_att = get_link(&mut rng,  BODIES, ATTRACTORS);
    let mut parent_pos = get_empty_vec2_simd(BODIES);
    let mut parent_mass = get_empty_vec_simd(BODIES);

    c.bench_function(
        "vec2 gravity par simd",
        |b| b.iter(|| {

            // parent_pos.get_from(&body_att, &att_pos);
            parent_pos.x.iter_mut()
                .zip(parent_pos.y.iter_mut())
                .zip(body_att.chunks_exact(8))
                .for_each(|((px, py), p)| {
                    p.iter()
                        .enumerate()
                        .for_each(|(i, parent)| {
                            if let (Some(p_pos_x), Some(p_pos_y)) = (att_pos.x.get(*parent), att_pos.y.get(*parent)) {
                                let _ = px.replace(i, *p_pos_x);
                                let _ = py.replace(i, *p_pos_y);
                            }
                        });
                });

            parent_mass.iter_mut()
                .zip(body_att.chunks_exact(8))
                .for_each(|(pm, p)| {
                    p.iter()
                        .enumerate()
                        .for_each(|(i, parent)| {
                            if let Some(p_mass) = att_mass.val.get(*parent) {
                                let _ = pm.replace(i, *p_mass);
                            }
                        });
                });

            parent_pos -= &body_pos;

            body_vel.x.par_iter_mut()
                .zip(body_vel.y.par_iter_mut())
                .zip(parent_pos.x.par_iter().zip(parent_pos.y.par_iter()))
                .zip(parent_mass.par_iter())
                .for_each(|(((bvx, bvy), (px, py)), pm)| {
                    let dist_squared = (*px * *px) + (*py * *py);
                    let dist = dist_squared.sqrt();
                    let ux = *px / dist;
                    let uy = *py / dist;
                    let a = *pm * G * DT / dist_squared;
                    *bvx += ux * a;
                    *bvy += uy * a;
                })
        })
    );
}

fn vec2_gravity_no_buffer(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let att_mass = get_vec(&mut rng, ATTRACTORS);
    let att_pos = get_vec2(&mut rng, ATTRACTORS);

    let body_pos = get_vec2(&mut rng, BODIES);
    let mut body_vel = get_vec2(&mut rng, BODIES);
    let body_att = get_link(&mut rng,  BODIES, ATTRACTORS);

    c.bench_function(
        "vec2 gravity no buffer",
        |b| b.iter(|| {
            body_vel.x.iter_mut()
                .zip(body_vel.y.iter_mut())
                .zip(body_pos.x.iter().zip(body_pos.y.iter()))
                .zip(body_att.iter())
                .for_each(|(((bvx, bvy), (bx, by)), p)| {
                    if let (Some(px), Some(py), Some(pm)) = (att_pos.x.get(*p), att_pos.y.get(*p), att_mass.val.get(*p)) {
                        let rel_x = *px - *bx;
                        let rel_y = *py - *by;
                        let dist_squared = (rel_x * rel_x) + (rel_y * rel_y);
                        let dist = dist_squared.sqrt();
                        let ux = *px / dist;
                        let uy = *py / dist;
                        let a = G * DT * *pm / dist_squared;
                        *bvx += ux * a;
                        *bvy += uy * a;
                    }
                })
        })
    );
}

fn vec2_gravity_par(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let att_mass = get_vec(&mut rng, ATTRACTORS);
    let att_pos = get_vec2(&mut rng, ATTRACTORS);

    let body_pos = get_vec2(&mut rng, BODIES);
    let mut body_vel = get_vec2(&mut rng, BODIES);
    let body_att = get_link(&mut rng,  BODIES, ATTRACTORS);
    let mut parent_pos = get_empty_vec2(BODIES);
    let mut parent_mass = get_empty_vec(BODIES);

    c.bench_function(
        "vec2 gravity par",
        |b| b.iter(|| {
            parent_pos.get_from(&body_att, &att_pos);
            parent_mass.get_from(&body_att, &att_mass);

            parent_pos -= &body_pos;

            body_vel.x.par_iter_mut()
                .zip(body_vel.y.par_iter_mut())
                .zip(parent_pos.x.par_iter().zip(parent_pos.y.par_iter()))
                .zip(parent_mass.val.par_iter())
                .for_each(|(((bvx, bvy), (px, py)), pm)| {
                    let dist_squared = (*px * *px) + (*py * *py);
                    let dist = dist_squared.sqrt();
                    let ux = *px / dist;
                    let uy = *py / dist;
                    let a = G * DT * *pm / dist_squared;
                    *bvx += ux * a;
                    *bvy += uy * a;
                })
        })
    );
}

fn get_vector(rng: &mut Rand64, n: usize) -> Vectors {
    Vectors {
        val: (0..n)
            .into_iter()
            .map(|_| Vector { x: rng.rand_float() as Float, y: rng.rand_float() as Float })
            .collect()
    }
}

fn vector_gravity(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let att_mass = get_vec(&mut rng, ATTRACTORS);
    let att_pos = get_vector(&mut rng, ATTRACTORS);

    let body_pos = get_vector(&mut rng, BODIES);
    let mut body_vel = get_vector(&mut rng, BODIES);
    let body_att = get_link(&mut rng,  BODIES, ATTRACTORS);

    c.bench_function(
        "vector gravity",
        |b| b.iter(|| {
            body_vel.val.iter_mut()
                .zip(body_pos.val.iter())
                .zip(body_att.iter())
                .filter_map(|((v, p), a)| {
                    let p_pos = *att_pos.val.get(*a)? - *p;
                    let pm = *att_mass.val.get(*a)?;
                    Some((v, p_pos, pm))
                })
                .for_each(|(bv, p_pos, pm)| {
                    let dist_squared = p_pos.magnitude_squared();
                    let dist = dist_squared.sqrt();
                    let u = p_pos / dist;
                    let a = G * DT * pm / dist_squared;
                    *bv += u * a;
                })
        })
    );
}

fn vector_gravity_par(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let att_mass = get_vec(&mut rng, ATTRACTORS);
    let att_pos = get_vector(&mut rng, ATTRACTORS);

    let body_pos = get_vector(&mut rng, BODIES);
    let mut body_vel = get_vector(&mut rng, BODIES);
    let body_att = get_link(&mut rng,  BODIES, ATTRACTORS);

    c.bench_function(
        "vector gravity par",
        |b| b.iter(|| {
            body_vel.val.par_iter_mut()
                .zip(body_pos.val.par_iter())
                .zip(body_att.par_iter())
                .filter_map(|((v, p), a)| {
                    let p_pos = *att_pos.val.get(*a)? - *p;
                    let pm = *att_mass.val.get(*a)?;
                    Some((v, p_pos, pm))
                })
                .for_each(|(bv, p_pos, pm)| {
                    let dist_squared = p_pos.magnitude_squared();
                    let a = G * DT * pm / dist_squared;
                    let u = p_pos / dist_squared.sqrt();
                    *bv += u * a;
                })
        })
    );
}

criterion_group!(
    movement,
    vec2_movement,
    vec2_movement_refined,
    vector_movement,
);

fn vec2_movement(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let mut pos = get_vec2(&mut rng, BODIES);
    let vel = get_vec2(&mut rng, BODIES);

    c.bench_function(
        "vec2 movement",
        |b| b.iter(|| {
            pos.x.iter_mut()
                .zip(vel.x.iter())
                .for_each(|(p, v)| {
                    *p += *v * DT;
                });

            pos.y.iter_mut()
                .zip(vel.y.iter())
                .for_each(|(p, v)| {
                    *p += *v * DT;
                });
        })
    );
}

fn vec2_movement_refined(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let mut pos = get_vec2(&mut rng, BODIES);
    let vel = get_vec2(&mut rng, BODIES);

    c.bench_function(
        "vec2 movement refined",
        |b| b.iter(|| pos += &vel * DT)
    );
}

fn vector_movement(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let mut pos = get_vector(&mut rng, BODIES);
    let vel = get_vector(&mut rng, BODIES);

    c.bench_function(
        "vector movement",
        |b| b.iter(|| {
            pos.val.iter_mut()
                .zip(vel.val.iter())
                .for_each(|(p, v)| {
                    *p += *v * DT;
                });
        })
    );
}

criterion_group!(
    add_assign_mul,
    vec2_add_assign_mul,
    vec2_add_assign_mul_inverted,
    vec2_add_assign_mul_refined,
    vec2_add_assign_mul_new,
    vector_add_assign_mul,
);

fn vec2_add_assign_mul(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let mut pos = get_vec2(&mut rng, BODIES);
    let vec2 = get_vec2(&mut rng, BODIES);
    let vec1 = get_vec(&mut rng, BODIES);

    c.bench_function(
        "vec2 add assign mul",
        |b| b.iter(|| {
            pos.x.iter_mut()
                .zip(pos.y.iter_mut())
                .zip(vec2.x.iter())
                .zip(vec2.y.iter())
                .zip(vec1.val.iter())
                .for_each(|((((x1, y1), x2), y2), n)| {
                    *x1 += *x2 * *n;
                    *y1 += *y2 * *n;
                });
        })
    );
}

fn vec2_add_assign_mul_inverted(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let mut pos = get_vec2(&mut rng, BODIES);
    let vec2 = get_vec2(&mut rng, BODIES);
    let vec1 = get_vec(&mut rng, BODIES);

    c.bench_function(
        "vec2 add assign mul inverted",
        |b| b.iter(|| {
            pos.x.iter_mut()
                .zip(pos.y.iter_mut())
                .zip(vec2.x.iter().zip(vec2.y.iter()))
                .zip(vec1.val.iter())
                .for_each(|(((x1, y1), (x2, y2)), n)| {
                    *x1 += *x2 * *n;
                    *y1 += *y2 * *n;
                });
        })
    );
}

fn vec2_add_assign_mul_refined(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let mut pos = get_vec2(&mut rng, BODIES);
    let vec2 = get_vec2(&mut rng, BODIES);
    let vec1 = get_vec(&mut rng, BODIES);

    c.bench_function(
        "vec2 add assign mul refined",
        |b| b.iter(|| {
            pos += &vec2 * &vec1;
        })
    );
}

fn get_new_vec1(rng: &mut Rand64, n: usize) -> simd_vecs::vecs::Vec1<Float> {
    get_vec(rng, n).val.into()
}

fn get_new_vec2(rng: &mut Rand64, n: usize) -> simd_vecs::vecs::Vec2<Float> {
    simd_vecs::vecs::Vec2 {
        x: get_vec(rng, n).val.into(),
        y: get_vec(rng, n).val.into(),
    }
}

fn vec2_add_assign_mul_new(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let mut pos = get_new_vec2(&mut rng, BODIES);
    let vec2 = get_new_vec2(&mut rng, BODIES);
    let vec1 = get_new_vec1(&mut rng, BODIES);

    c.bench_function(
        "vec2 add assign mul new",
        |b| b.iter(|| {
            pos += &vec2 * &vec1;
        })
    );
}

fn vector_add_assign_mul(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let mut pos = get_vector(&mut rng, BODIES);
    let vec2 = get_vector(&mut rng, BODIES);
    let vec1 = get_vec(&mut rng, BODIES);

    c.bench_function(
        "vector add assign mul",
        |b| b.iter(|| {
            pos.val.iter_mut()
                .zip(vec2.val.iter())
                .zip(vec1.val.iter())
                .for_each(|((a, b), c)| {
                    *a += *b * *c;
                })
        })
    );
}
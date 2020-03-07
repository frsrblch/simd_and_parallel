use criterion::*;
use oorandom::Rand64;
use simd_and_parallel::*;

const N: usize = 50_000;

fn get_floats_vec() -> Vec<(Float, Float)> {
    (0..).into_iter()
        .map(|i| (i as Float * 0.9, i as Float * 1.1))
        .take(N)
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

const ATTRACTORS: usize = 100;
const BODIES: usize = 5000;

fn get_vec2(rng: &mut Rand64, n: usize) -> Vec2 {
    (0..n)
        .into_iter()
        .fold(Vec2::default(), |mut v, _| {
            v.x.push(rng.rand_float() as Float);
            v.y.push(rng.rand_float() as Float);
            v
        })
}

fn get_vec(rng: &mut Rand64, n: usize) -> Vec1 {
    Vec1 {
        val: (0..n)
            .into_iter()
            .map(|_| rng.rand_float() as Float)
            .collect()
    }
}

fn get_link(rng: &mut Rand64, n: usize, range: usize) -> Vec<usize> {
    (0..n)
        .into_iter()
        .map(|_| rng.rand_range(0..range as u64) as usize)
        .collect()
}

fn get_empty_vec2(n: usize) -> Vec2 {
    Vec2 {
        x: (0..n).into_iter().map(|_| 0.0).collect(),
        y: (0..n).into_iter().map(|_| 0.0).collect(),
    }
}

fn get_empty_vec(n: usize) -> Vec1 {
    Vec1 {
        val: (0..n).into_iter().map(|_| 0.0).collect()
    }
}

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
            // GET PARENT POSITION
            parent_pos.get_from(&body_att, &att_pos);

            // GET PARENT MASS
            parent_mass.get_from(&body_att, &att_mass);

            // CALCULATE FORCE DUE TO GRAVITY
            parent_pos -= &body_pos;

            body_vel.x.iter_mut()
                .zip(body_vel.y.iter_mut())
                .zip(parent_pos.x.iter())
                .zip(parent_pos.y.iter())
                .zip(parent_mass.val.iter())
                .for_each(|((((bvx, bvy), px), py), pm)| {
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
                // .zip(parent_pos.val.iter())
                // .zip(parent_mass.iter())
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

criterion_group!(
    add_assign,
    vectors_add_assign,
    vec2_add_assign,
);

criterion_group!(
    gravity,
    vec2_gravity,
    vector_gravity,
);

criterion_main! {
    add_assign,
    gravity,
}
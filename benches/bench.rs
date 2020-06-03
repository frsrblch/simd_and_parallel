use criterion::*;
use oorandom::Rand64;
use simd_and_parallel::*;
use rayon::prelude::*;

const ATTRACTORS: usize = 100;
const BODIES: usize = 4096*16;
const MOONS: usize = 4096;

const LANES: usize = SimdVector::lanes();

fn random_simd_vector() -> SimdVector {
    f32x8::new(random(), random(), random(), random(), random(), random(), random(), random())
}

criterion_main! {
    add_assign,
    gravity,
    movement,
    add_assign_mul,
    moon_orbit,
}

criterion_group!(
    moon_orbit,
    vector_orbits,
    vec_orbits,
    simd_orbits,
    vector_orbits_simple,
);

struct OrbitParams {
    parent: usize,
    id: usize,
    period: Float,
    radius: Float,
    offset: Float,
}

fn vector_orbits(c: &mut Criterion) {
    let rng = &mut Rand64::new(0);

    let mut body_position = get_vector(rng, MOONS);

    let mut body_index = vec![];

    let mut planet = 0;
    for i in 0..MOONS {
        // make every fourth body a planet
        if i % 4 == 0 {
            planet = i;
        } else {
            body_index.push(OrbitParams {
                parent: planet,
                id: i,
                radius: rng.rand_float() as Float,
                period: rng.rand_float() as Float,
                offset: rng.rand_float() as Float,
            });
        }
    }

    let time = rng.rand_float() as Float;

    c.bench_function(
        "vector orbits",
        |b| b.iter(|| {
            body_index
                .iter()
                .for_each(|orbit| {
                    let orbit_fraction = time / orbit.period + orbit.offset;
                    let angle = orbit_fraction * 2.0 * PI as Float;
                    let x = orbit.radius * angle.cos();
                    let y = orbit.radius * angle.sin();

                    if let Some(&planet_pos) = body_position.val.get(orbit.parent) {
                        if let Some(body_position) = body_position.val.get_mut(orbit.id) {
                            body_position.x = planet_pos.x + x;
                            body_position.y = planet_pos.y + y;
                        }
                    }
                })
        })
    );
}

fn vec_orbits(c: &mut Criterion) {
    let rng = &mut Rand64::new(0);

    let mut body_position = get_vec2(rng, MOONS);

    let mut planet_index = vec![];
    let mut body_index = vec![];

    let mut planet = 0;
    for i in 0..MOONS {
        // make every fourth body a planet
        if i % 4 == 0 {
            planet = i;
        } else {
            planet_index.push(planet);
            body_index.push(i);
        }
    }

    let mut parent_position = get_empty_vec2(planet_index.len());
    let mut moon_position = get_empty_vec2(planet_index.len());
    let orbit_period = get_vec(rng, planet_index.len());
    let orbit_radius = get_vec(rng, planet_index.len());
    let orbit_offset = get_vec(rng, planet_index.len());

    let time = rng.rand_float() as Float;

    c.bench_function(
        "vec orbits",
        |b| b.iter(|| {
            parent_position.get_from(&planet_index, &body_position);

            moon_position.x.iter_mut()
                .zip(moon_position.y.iter_mut())
                .zip(parent_position.x.iter())
                .zip(parent_position.y.iter())
                .zip(orbit_period.val.iter())
                .zip(orbit_radius.val.iter())
                .zip(orbit_offset.val.iter())
                .for_each(|((((((mx, my), &px), &py), &period), &radius), &offset)| {
                    let orbit_fraction = time / period + offset;
                    let angle = orbit_fraction * 2.0 * PI as Float;
                    let x = radius * angle.cos();
                    let y = radius * angle.sin();

                    *mx = x + px;
                    *my = y + py;
                });

            body_index.iter()
                .zip(moon_position.x.iter())
                .zip(moon_position.y.iter())
                .for_each(|((&i, mx), my)| {
                    if let (Some(x), Some(y)) = (body_position.x.get_mut(i), body_position.y.get_mut(i)) {
                        *x = *mx;
                        *y = *my;
                    }
                });
        })
    );
}

struct SimdOrbit {
    period: SimdVector,
    radius: SimdVector,
    offset: SimdVector,
}

impl SimdOrbit {
    fn random() -> Self {
        Self {
            period: random_simd_vector(),
            radius: random_simd_vector(),
            offset: random_simd_vector(),
        }
    }
}

use rand::random;

fn simd_orbits(c: &mut Criterion) {
    let rng = &mut Rand64::new(0);

    let mut body_position = get_vec2(rng, MOONS);

    let mut planet_index = vec![];
    let mut body_index = vec![];

    let mut planet = 0;
    for i in 0..MOONS {
        // make every fourth body a planet
        if i % 4 == 0 {
            planet = i;
        } else {
            planet_index.push(planet);
            body_index.push(i);
        }
    }

    let mut parent_position = get_empty_vec2_simd(planet_index.len());
    let mut moon_position = get_empty_vec2_simd(planet_index.len());

    let orbit = (0..planet_index.len() / LANES)
        .into_iter()
        .map(|_| SimdOrbit::random())
        .collect::<Vec<_>>();

    let time = random();

    c.bench_function(
        "simd orbits",
        |b| b.iter(|| {
            parent_position.x.iter_mut()
                .zip(parent_position.y.iter_mut())
                .zip(planet_index.iter().as_slice().chunks_exact(LANES))
                .for_each(|((x, y), ids)| {
                    ids.iter()
                        .enumerate()
                        .for_each(|(i, body)| {
                            *x = x.replace(i, *body_position.x.get(*body).unwrap_or(&0.0));
                            *y = y.replace(i, *body_position.y.get(*body).unwrap_or(&0.0));
                        });
                });

            moon_position.x.iter_mut()
                .zip(moon_position.y.iter_mut())
                .zip(parent_position.x.iter())
                .zip(parent_position.y.iter())
                .zip(orbit.iter())
                .for_each(|((((mx, my), &px), &py), orbit)| {
                    let orbit_fraction = SimdVector::splat(time) / orbit.period + orbit.offset;
                    let angle = orbit_fraction * 2.0 * PI as Float;

                    let x = orbit.radius * angle.cos();
                    let y = orbit.radius * angle.sin();

                    *mx = x + px;
                    *my = y + py;
                });

            body_index.iter().as_slice().chunks_exact(LANES)
                .zip(moon_position.x.iter())
                .zip(moon_position.y.iter())
                .for_each(|((ids, mx), my)| {
                    ids.iter()
                        .enumerate()
                        .for_each(|(i, body)| {
                            if let Some(x) = body_position.x.get_mut(*body) {
                                *x = mx.extract(i);
                            }
                            if let Some(y) = body_position.y.get_mut(*body) {
                                *y = my.extract(i);
                            }
                        });
                });
        })
    );
}

pub struct PlanetOrbit {
    pub radius: Float,
    pub period: Float,
    pub offset: Float,
}

impl PlanetOrbit {
    pub fn random() -> Self {
        Self {
            radius: random(),
            period: random(),
            offset: random(),
        }
    }
}

pub struct MoonOrbit {
    pub parent: usize,
    pub radius: Float,
    pub period: Float,
    pub offset: Float,
}

impl MoonOrbit {
    pub fn random(parent: usize) -> Self {
        Self {
            parent,
            radius: random(),
            period: random(),
            offset: random(),
        }
    }
}

fn vector_orbits_simple(c: &mut Criterion) {
    let mut bodies = (0..MOONS).into_iter()
        .map(|_| Body::random(ATTRACTORS))
        .collect::<Vec<_>>();

    let mut orbits = std::collections::HashMap::<usize, PlanetOrbit>::default();
    let mut moon_orbits = std::collections::HashMap::<usize, MoonOrbit>::default();

    let mut planet = 0;
    for i in 0..MOONS {
        if i % 4 == 0 {
            orbits.insert(i, PlanetOrbit::random());
            planet = i;
        } else {
            moon_orbits.insert(i, MoonOrbit::random(planet));
        }
    }

    let time = random::<Float>();

    c.bench_function(
        "planet orbits simple",
        |b| b.iter(|| {
            orbits.iter()
                .for_each(|(body, orbit)| {
                    if let Some(body) = bodies.get_mut(*body) {
                        let orbit_fraction = time / orbit.period + orbit.offset;
                        let angle = orbit_fraction * 2.0 * PI as Float;

                        body.position.x = orbit.radius * angle.cos();
                        body.position.y = orbit.radius * angle.sin();
                    }
                });

            moon_orbits.iter()
                .for_each(|(body, orbit)| {
                    let parent = bodies.get(orbit.parent);
                    if let Some(parent_pos) = parent.map(|p| p.position) {
                        if let Some(body) = bodies.get_mut(*body) {
                            let orbit_fraction = time / orbit.period + orbit.offset;
                            let angle = orbit_fraction * 2.0 * PI as Float;

                            body.position.x = parent_pos.x + orbit.radius * angle.cos();
                            body.position.y = parent_pos.y + orbit.radius * angle.sin();
                        }
                    }
                });
        })
    );
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

fn get_vec2_simd(n: usize) -> Vec2<SimdVector> {
    assert_eq!(0, n % LANES);

    (0..n/LANES)
        .into_iter()
        .fold(Vec2::default(), |mut v, _| {
            v.x.push(random_simd_vector());
            v.y.push(random_simd_vector());
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

fn get_empty_vec2_simd(n: usize) -> Vec2<SimdVector> {
    assert_eq!(0, n % LANES);

    Vec2 {
        x: (0..n/LANES).into_iter().map(|_| Default::default()).collect(),
        y: (0..n/LANES).into_iter().map(|_| Default::default()).collect(),
    }
}

fn get_empty_vec(n: usize) -> Vec1<Float> {
    Vec1 {
        val: (0..n).into_iter().map(|_| 0.0).collect()
    }
}

fn get_empty_vec_simd(n: usize) -> Vec<SimdVector> {
    assert_eq!(0, n % LANES);

    (0..n).into_iter().map(|_| Default::default()).collect()
}

criterion_group!(
    gravity,
    vec2_gravity,
    vec2_gravity_simd,
    vec2_gravity_par,
    vector_gravity,
    vector_gravity_simple,
    vector_gravity_par,
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

            body_vel.iter_mut()
                .zip(parent_pos.iter())
                .zip(parent_mass.val.iter())
                .for_each(|(((bvx, bvy), (px, py)), pm)| {
                    let dist_squared = (*px * *px) + (*py * *py);
                    let dist = dist_squared.sqrt();
                    let ux = *px;
                    let uy = *py;
                    let a = G * DT * *pm / dist_squared / dist;
                    *bvx += ux * a;
                    *bvy += uy * a;
                })
        })
    );
}

use packed_simd::*;
use std::f64::consts::PI;

fn vec2_gravity_simd(c: &mut Criterion) {
    let mut rng = Rand64::new(0);

    let att_mass = get_vec(&mut rng, ATTRACTORS);
    let att_pos = get_vec2(&mut rng, ATTRACTORS);

    let body_pos = get_vec2(&mut rng, BODIES);
    let mut body_vel = get_vec2(&mut rng, BODIES);
    let body_att = get_link(&mut rng,  BODIES, ATTRACTORS);
    let mut parent_pos = get_empty_vec2(BODIES);
    let mut parent_mass = get_empty_vec(BODIES);

    c.bench_function(
        "vec2 gravity simd",
        |b| b.iter(|| {
            parent_pos.get_from(&body_att, &att_pos);
            parent_mass.get_from(&body_att, &att_mass);

            parent_pos -= &body_pos;

            body_vel.x.chunks_exact_mut(LANES)
                .zip(body_vel.y.chunks_exact_mut(LANES))
                .zip(parent_pos.x.chunks_exact(LANES))
                .zip(parent_pos.y.chunks_exact(LANES))
                .zip(parent_mass.val.chunks_exact(LANES))
                .for_each(|((((bvx, bvy), px), py), pm)| {
                    let px = SimdVector::from_slice_unaligned(px);
                    let py = SimdVector::from_slice_unaligned(py);

                    let dist_squared = (px * px) + (py * py);
                    let dist = dist_squared.sqrte();

                    let pm = SimdVector::from_slice_unaligned(pm);
                    let a = G * DT * pm / dist_squared / dist;

                    let mut vx = SimdVector::from_slice_unaligned(bvx);
                    let mut vy = SimdVector::from_slice_unaligned(bvy);

                    vx += px * a;
                    vy += py * a;

                    vx.write_to_slice_unaligned(bvx);
                    vy.write_to_slice_unaligned(bvy);
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

    c.bench_function(
        "vec2 gravity par",
        |b| b.iter(|| {

            body_vel.x.par_iter_mut()
                .zip(body_vel.y.par_iter_mut())
                .zip(body_pos.x.par_iter())
                .zip(body_pos.y.par_iter())
                .zip(body_att.par_iter())
                .for_each(|((((vx, vy), bx), by), &a)| {
                    if let (Some(px), Some(py), Some(pm)) = (att_pos.x.get(a), att_pos.y.get(a), att_mass.val.get(a)) {
                        let rel_x = *px - *bx;
                        let rel_y = *py - *by;

                        let dist_squared = rel_x * rel_x + rel_y * rel_y;
                        let dist = dist_squared.sqrt();

                        let a = G * DT * *pm / dist_squared / dist;

                        *vx += rel_x * a;
                        *vy += rel_y * a;
                    }
                });
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
                    let p_pos = att_pos.val.get(*a)?;
                    let pm = *att_mass.val.get(*a)?;
                    Some((v, p_pos, p, pm))
                })
                .for_each(|(v, pp, bp, pm)| {
                    let rel_x = pp.x - bp.x;
                    let rel_y = pp.y - bp.y;

                    let distance_squared = rel_x * rel_x + rel_y * rel_y;

                    let distance = distance_squared.sqrt();
                    let a = G * DT * pm / distance_squared / distance;

                    v.x += rel_x * a;
                    v.y += rel_y * a;
                })
        })
    );
}

fn vector_gravity_simple(c: &mut Criterion) {
    let attractors = (0..ATTRACTORS).into_iter()
        .map(|_| Parent::random())
        .collect::<Vec<_>>();

    let mut bodies = (0..BODIES).into_iter()
        .map(|_| Body::random(ATTRACTORS))
        .collect::<Vec<_>>();

    c.bench_function(
        "vector gravity simple",
        |b| b.iter(|| {
            bodies.iter_mut()
                .for_each(|body| {
                    if let Some(parent) = attractors.get(body.attractor) {
                        let rx = parent.position.x - body.position.x;
                        let ry = parent.position.y - body.position.y;
                        let distance_squared = rx * rx + ry * ry;
                        let distance = distance_squared.sqrt();
                        let a = G * DT * parent.mass / distance_squared / distance;
                        body.velocity.x += rx * a;
                        body.velocity.y += ry * a;
                    }
                });
        })
    );
}

struct Body {
    pub attractor: usize,
    pub position: Vector,
    pub velocity: Vector,
}

impl Body {
    pub fn random(n_att: usize) -> Self {
        Body {
            attractor: random::<usize>() % n_att,
            position: Vector::random(),
            velocity: Vector::random(),
        }
    }
}

struct Parent {
    mass: Float,
    position: Vector,
}

impl Parent {
    pub fn random() -> Self {
        Parent {
            mass: random(),
            position: Vector::random(),
        }
    }
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
                    let p_pos = att_pos.val.get(*a)?;
                    let pm = *att_mass.val.get(*a)?;
                    Some((v, p_pos, p, pm))
                })
                .for_each(|(v, pp, bp, pm)| {
                    let rel_x = pp.x - bp.x;
                    let rel_y = pp.y - bp.y;

                    let distance_squared = rel_x * rel_x + rel_y * rel_y;

                    let distance = distance_squared.sqrt();
                    let a = G * DT * pm / distance_squared / distance;

                    v.x += rel_x * a;
                    v.y += rel_y * a;
                })
        })
    );
}

pub struct BodySimd {
    pub attractor: [usize; LANES],
    pub pos_x: SimdVector,
    pub pos_y: SimdVector,
    pub vel_x: SimdVector,
    pub vel_y: SimdVector,
}

impl BodySimd {
    pub fn random(n_att: usize) -> Self {
        Self {
            attractor: Self::get_att(n_att),
            pos_x: random_simd_vector(),
            pos_y: random_simd_vector(),
            vel_x: random_simd_vector(),
            vel_y: random_simd_vector(),
        }
    }

    fn get_att(n_att: usize) -> [usize; LANES] {
        [
            random::<usize>() % n_att,
            random::<usize>() % n_att,
            random::<usize>() % n_att,
            random::<usize>() % n_att,
            random::<usize>() % n_att,
            random::<usize>() % n_att,
            random::<usize>() % n_att,
            random::<usize>() % n_att
        ]
    }
}

pub struct AttractorSimd {
    pub pos_x: SimdVector,
    pub pos_y: SimdVector,
    pub mass: SimdVector,
}

impl AttractorSimd {
    pub fn random() -> Self {
        Self {
            pos_x: random_simd_vector(),
            pos_y: random_simd_vector(),
            mass: random_simd_vector(),
        }
    }
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
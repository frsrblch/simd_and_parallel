use rayon::prelude::*;
pub use packed_simd::f64x2;


pub fn magnitude_squared(x: &[f64], y: &[f64], m: &mut [f64]) {
//    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//    {
//        if is_x86_feature_detected!("avx2") {
//            return unsafe { magnitude_squared_avx2(x, y, m); }
//        }
//    }

    return magnitude_squared_fallback(x, y, m);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn magnitude_squared_avx2(x: &[f64], y: &[f64], m: &mut [f64]) {
    magnitude_squared_fallback(x, y ,m);
}

pub fn magnitude_squared_fallback(x: &[f64], y: &[f64], m: &mut [f64]) {
    m.iter_mut()
        .zip(x.iter())
        .zip(y.iter())
        .for_each(|((m, x),  y)| {
            *m = *x * *x + *y * *y;
        })
}

pub fn parallel_magnitude_squared_fallback(x: &[f64], y: &[f64], m: &mut [f64]) {
    m.par_iter_mut()
        .zip(x.par_iter())
        .zip(y.par_iter())
        .for_each(|((m, x), y)| {
            *m = *x * *x + *y * *y;
        })
}

pub fn parallel_magnitude_squared(x: &[f64], y: &[f64], m: &mut [f64]) {
    m.par_chunks_mut(256)
        .zip(x.par_chunks(256))
        .zip(y.par_chunks(256))
        .for_each(|((m, x), y)| {
            magnitude_squared(x, y, m);
        })
}

pub fn packed_magnitude_squared(v: &[f64x2], m: &mut [f64]) {
    m.iter_mut()
        .zip(v.iter())
        .for_each(|(m, v)| {
            *m = (*v * *v).sum();
        })
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Vector {
    pub x: f64,
    pub y: f64,
}

impl Vector {
    pub fn magnitude_squared(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn msqrd(v: &[Vector], m: &mut [f64]) {
        v.iter()
            .zip(m.iter_mut())
            .for_each(|(v, m)| {
                *m = v.magnitude_squared();
            });
    }

    pub fn par_msqrd(v: &[Vector], m: &mut [f64]) {
        v.par_iter()
            .zip(m.par_iter_mut())
            .for_each(|(v, m)| {
                *m = v.magnitude_squared();
            })
    }
}
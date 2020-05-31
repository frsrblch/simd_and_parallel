#[macro_use] extern crate itertools;
use std::ops::*;
use crate::types::VMul;

pub mod types;

pub type Float = f32;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Vec1<T> {
    pub val: Vec<T>,
}

impl<T: Copy> Vec1<T> {
    pub fn get_from(&mut self, indices: &Vec<usize>, values: &Vec1<T>) {
        self.val.iter_mut()
            .zip(indices.iter())
            .for_each(|(v, i)| {
                if let Some(val) = values.val.get(*i) {
                    *v = *val;
                }
            });
    }
}

impl Vec1<f32> {
    pub fn get_magnitude(&mut self, vec: &Vec2<f32>) {
        self.val.iter_mut()
            .zip(vec.x.iter())
            .zip(vec.y.iter())
            .for_each(|((v, x), y)| {
                *v = ((*x * *x) + (*y * *y)).sqrt();
            })
    }
}

impl Vec1<f64> {
    pub fn get_magnitude(&mut self, vec: &Vec2<f64>) {
        self.val.iter_mut()
            .zip(vec.x.iter())
            .zip(vec.y.iter())
            .for_each(|((v, x), y)| {
                *v = ((*x * *x) + (*y * *y)).sqrt();
            })
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Vec2<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
}

impl<'a, T> Mul<T> for &'a Vec2<T> {
    type Output = VMul<&'a Vec2<T>, T>;

    fn mul(self, rhs: T) -> Self::Output {
        types::VMul(self, rhs)
    }
}

impl<'a, T> Mul<&'a Vec1<T>> for &'a Vec2<T> {
    type Output = VMul<&'a Vec2<T>, &'a Vec1<T>>;

    fn mul(self, rhs: &'a Vec1<T>) -> Self::Output {
        types::VMul(self, rhs)
    }
}

impl<'a, T: Copy + Mul<Output=T> + AddAssign> AddAssign<VMul<&'a Self, T>> for Vec2<T> {
    fn add_assign(&mut self, rhs: VMul<&'a Vec2<T>, T>) {
        self.x.iter_mut()
            .zip(self.y.iter_mut())
            .zip(rhs.0.x.iter())
            .zip(rhs.0.y.iter())
            .for_each(|(((x1, y1), x2), y2)| {
                *x1 += *x2 * rhs.1;
                *y1 += *y2 * rhs.1;
            })
    }
}

impl<'a, T: Copy + Mul<Output=T> + AddAssign> AddAssign<VMul<&'a Self, &'a Vec1<T>>> for Vec2<T> {
    fn add_assign(&mut self, rhs: VMul<&'a Vec2<T>, &'a Vec1<T>>) {
        self.x.iter_mut()
            .zip(self.y.iter_mut())
            .zip(rhs.0.x.iter())
            .zip(rhs.0.y.iter())
            .zip(rhs.1.val.iter())
            .for_each(|((((x1, y1), x2), y2), x)| {
                *x1 += *x2 * *x;
                *y1 += *y2 * *x;
            })
    }
}

impl<T: Copy> Vec2<T> {
    pub fn get_from(&mut self, indices: &Vec<usize>, values: &Vec2<T>) {
        self.x.iter_mut()
            .zip(self.y.iter_mut())
            .zip(indices.iter())
            .for_each(|((x, y), i)| {
                if let Some(val) = values.x.get(*i) {
                    *x = *val;
                }
                if let Some(val) = values.y.get(*i) {
                    *y = *val;
                }
            });
    }
}

impl<T: Copy + AddAssign + Mul<Output=T>> AddAssign<(&Self, T)> for Vec2<T> {
    fn add_assign(&mut self, (rhs, f): (&Vec2<T>, T)) {
        self.x.iter_mut()
            .zip(self.y.iter_mut())
            .zip(rhs.x.iter())
            .zip(rhs.y.iter())
            .for_each(|(((x1, y1), x2), y2)| {
                *x1 += *x2 * f;
                *y1 += *y2 * f;
            });
    }
}

impl<T: Copy + AddAssign> AddAssign<&Self> for Vec2<T> {
    fn add_assign(&mut self, rhs: &Vec2<T>) {
        self.x.iter_mut()
            .zip(rhs.x.iter())
            .for_each(|(s, r)| {
                *s += *r;
            });

        self.y.iter_mut()
            .zip(rhs.y.iter())
            .for_each(|(s, r)| {
                *s += *r;
            });
    }
}

impl<T: Copy + SubAssign> SubAssign<&Self> for Vec2<T> {
    fn sub_assign(&mut self, rhs: &Vec2<T>) {
        self.x.iter_mut()
            .zip(rhs.x.iter())
            .for_each(|(s, r)| {
                *s -= *r;
            });

        self.y.iter_mut()
            .zip(rhs.y.iter())
            .for_each(|(s, r)| {
                *s -= *r;
            });
    }
}

impl<T: Copy + MulAssign> MulAssign<&Vec1<T>> for Vec2<T> {
    fn mul_assign(&mut self, rhs: &Vec1<T>) {
        self.x.iter_mut()
            .zip(rhs.val.iter())
            .for_each(|(s, r)| {
                *s *= *r;
            });

        self.y.iter_mut()
            .zip(rhs.val.iter())
            .for_each(|(s, r)| {
                *s *= *r;
            });
    }
}

impl<T: Copy + DivAssign> DivAssign<&Vec1<T>> for Vec2<T> {
    fn div_assign(&mut self, rhs: &Vec1<T>) {
        self.x.iter_mut()
            .zip(rhs.val.iter())
            .for_each(|(s, r)| {
                *s /= *r;
            });

        self.y.iter_mut()
            .zip(rhs.val.iter())
            .for_each(|(s, r)| {
                *s /= *r;
            });
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Vector {
    pub x: Float,
    pub y: Float,
}

impl Vector {
    pub fn magnitude_squared(self) -> Float {
        self.x * self.x + self.y * self.y
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Sub for Vector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector {
            x: self.x - rhs.x,
            y: self.y - rhs.y
        }
    }
}

impl Mul<Float> for Vector {
    type Output = Self;

    fn mul(self, rhs: Float) -> Self::Output {
        Vector {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Div<Float> for Vector {
    type Output = Self;

    fn div(self, rhs: Float) -> Self::Output {
        Vector {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Vectors {
    pub val: Vec<Vector>,
}

impl Vectors {
    pub fn get_from(&mut self, indices: &Vec<usize>, values: &Vectors) {
        self.val.iter_mut()
            .zip(indices.iter())
            .for_each(|(v, i)| {
                if let Some(val) = values.val.get(*i) {
                    *v = *val;
                }
            });
    }
}

impl AddAssign<&Self> for Vectors {
    fn add_assign(&mut self, rhs: &Vectors) {
        self.val.iter_mut()
            .zip(rhs.val.iter())
            .for_each(|(s, r)| {
                *s += *r;
            })
    }
}

impl SubAssign<&Self> for Vectors {
    fn sub_assign(&mut self, rhs: &Vectors) {
        self.val.iter_mut()
            .zip(rhs.val.iter())
            .for_each(|(s, r)| {
                *s -= *r;
            })
    }
}

#[derive(Debug, Default, Clone)]
pub struct Vec3 {
    pub x: Vec<Float>,
    pub y: Vec<Float>,
    pub z: Vec<Float>,
}

use std::iter::once;
impl Vec3 {
    fn iter_mut(&mut self) -> impl Iterator<Item=&mut [Float]> {
        once(self.x.as_mut_slice())
            .chain(once(self.y.as_mut_slice()))
            .chain(once(self.z.as_mut_slice()))
    }

    fn iter(&self) -> impl Iterator<Item=&[Float]> {
        once(self.x.as_slice())
            .chain(once(self.y.as_slice()))
            .chain(once(self.z.as_slice()))
    }

    fn do_for_all<F: Fn(&mut Float, Float)>(&mut self, rhs: &Vec3, f: F) {
        self.iter_mut()
            .zip(rhs.iter())
            .flat_map(|(l, r)| l.iter_mut().zip(r.iter()))
            .for_each(|(l, r)| f(l, *r));
    }

    fn do_for_each<F: Fn(&mut Float, Float)>(&mut self, rhs: Float, f: F) {
        self.iter_mut()
            .flat_map(|values| values.iter_mut())
            .for_each(|value| f(value, rhs));
    }
}

impl AddAssign<&Self> for Vec3 {
    fn add_assign(&mut self, rhs: &Vec3) {
        self.do_for_all(rhs, Float::add_assign);
    }
}

impl AddAssign<Float> for Vec3 {
    fn add_assign(&mut self, rhs: Float) {
        self.do_for_each(rhs, Float::add_assign);
    }
}

impl SubAssign<&Self> for Vec3 {
    fn sub_assign(&mut self, rhs: &Vec3) {
        self.do_for_all(rhs, Float::sub_assign);
    }
}

impl SubAssign<Float> for Vec3 {
    fn sub_assign(&mut self, rhs: Float) {
        self.do_for_each(rhs, Float::sub_assign);
    }
}

impl MulAssign<&Self> for Vec3 {
    fn mul_assign(&mut self, rhs: &Vec3) {
        self.do_for_all(rhs, Float::mul_assign);
    }
}

impl MulAssign<Float> for Vec3 {
    fn mul_assign(&mut self, rhs: Float) {
        self.do_for_each(rhs, Float::mul_assign);
    }
}

impl DivAssign<&Self> for Vec3 {
    fn div_assign(&mut self, rhs: &Vec3) {
        self.do_for_all(rhs, Float::div_assign);
    }
}

impl DivAssign<Float> for Vec3 {
    fn div_assign(&mut self, rhs: Float) {
        self.do_for_each(rhs, Float::div_assign);
    }
}

#[derive(Debug, Default, Clone)]
pub struct Vectors3 {
    pub val: Vec<Vector3>,
}

impl AddAssign<&Self> for Vectors3 {
    fn add_assign(&mut self, rhs: &Vectors3) {
        izip!(self.val.iter_mut(), rhs.val.iter())
            .for_each(|(lhs, rhs)| {
                *lhs += *rhs;
            })
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Vector3 {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

impl AddAssign for Vector3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

#[test]
fn add_assign_test() {
    let mut a = Vec2 {
        x: vec![0.0, 1.0, 2.0],
        y: vec![0.0, 2.0, 3.0],
    };

    let b = Vec2 {
        x: vec![1.0, 1.0, 1.0],
        y: vec![1.0, 1.0, 1.0],
    };

    let expected = Vec2 {
        x: vec![1.0, 2.0, 3.0],
        y: vec![1.0, 3.0, 4.0],
    };

    a += &b;

    assert_eq!(expected, a);
}

#[test]
fn mul_assign_test() {
    let mut a = Vec2 {
        x: vec![0.0, 1.0, 2.0],
        y: vec![0.0, 2.0, 3.0],
    };

    let b = Vec1 {
        val: vec![2.0, 3.0, 5.0],
    };

    let expected = Vec2 {
        x: vec![0.0, 3.0, 10.0],
        y: vec![0.0, 6.0, 15.0],
    };

    a *= &b;

    assert_eq!(expected, a);
}
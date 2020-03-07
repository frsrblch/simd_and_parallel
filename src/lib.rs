use std::ops::*;

pub type Float = f32;

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

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Vec1 {
    pub val: Vec<Float>,
}

impl Vec1 {
    pub fn get_from(&mut self, indices: &Vec<usize>, values: &Vec1) {
        self.val.iter_mut()
            .zip(indices.iter())
            .for_each(|(v, i)| {
                if let Some(val) = values.val.get(*i) {
                    *v = *val;
                }
            });
    }

    pub fn get_magnitude(&mut self, vec: &Vec2) {
        self.val.iter_mut()
            .zip(vec.x.iter())
            .zip(vec.y.iter())
            .for_each(|((v, x), y)| {
                *v = ((*x * *x) + (*y * *y)).sqrt();
            })
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Vec2 {
    pub x: Vec<Float>,
    pub y: Vec<Float>,
}

impl Vec2 {
    pub fn get_from(&mut self, indices: &Vec<usize>, values: &Vec2) {
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

impl AddAssign<(&Self, Float)> for Vec2 {
    fn add_assign(&mut self, (rhs, f): (&Vec2, Float)) {
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

impl AddAssign<&Self> for Vec2 {
    fn add_assign(&mut self, rhs: &Vec2) {
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

impl SubAssign<&Self> for Vec2 {
    fn sub_assign(&mut self, rhs: &Vec2) {
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

impl MulAssign<&Vec1> for Vec2 {
    fn mul_assign(&mut self, rhs: &Vec1) {
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

impl DivAssign<&Vec1> for Vec2 {
    fn div_assign(&mut self, rhs: &Vec1) {
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
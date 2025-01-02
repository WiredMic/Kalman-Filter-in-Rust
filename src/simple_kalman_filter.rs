#![allow(unused_imports)]
#![allow(dead_code)]

// gaussian struct
// with addition and multiplication

use core::ops::{Add, Mul};

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    mean: f32,
    variance: f32,
}

impl Gaussian {
    pub fn new(mean: f32, variance: f32) -> Self {
        Gaussian { mean, variance }
    }

    pub fn get_fields(&self) -> (f32, f32) {
        (self.mean, self.variance)
    }
}

// \[\mu_{3} = \mu_{1} + \mu_{2}\]
// \[\sigma_{3}^{2} = \sigma_{1}^{2} + \sigma_{2}^{2} \]
impl Add for Gaussian {
    type Output = Gaussian;
    fn add(self: Gaussian, b: Gaussian) -> Gaussian {
        let mean3 = self.mean + b.mean;
        let variance3 = self.variance + b.variance;
        Gaussian::new(mean3, variance3)
    }
}

// \[\mu _{3} = \frac{\sigma_{1}^{2}\mu_{2}+\sigma_{2}^{2}\mu_{1} }{\sigma_{1}^{2}+\sigma_{2}^{2} } \]
// \[\sigma_{3}^{2 }= \frac{\sigma_{1}^{2}\sigma_{2}^{2}}{\sigma_{1}^{2}+\sigma_{2}^{2}}\]
impl Mul for Gaussian {
    type Output = Gaussian;
    fn mul(self: Gaussian, b: Gaussian) -> Gaussian {
        let mean3 =
            (self.variance * b.mean + b.variance * self.mean) / (self.variance + b.variance);
        let variance3 = (self.variance * b.variance) / (self.variance + b.variance);

        Gaussian::new(mean3, variance3)
    }
}

// // scalar/multivector wedge
// impl Mul<Gaussian> for f32 {
//     type Output = Gaussian;

//     fn mul(self: f32, b: Gaussian) -> Gaussian {
//         Gaussian::new(self * b.mean, b.variance)
//     }
// }

// // multivector/scalar wedge
// impl Mul<f32> for Gaussian {
//     type Output = Gaussian;

//     fn mul(self: Gaussian, scalar: f32) -> Gaussian {
//         Gaussian::new(self.mean * scalar, self.variance)
//     }
// }

pub struct SimpleKalman {
    x: Gaussian,
    fx: Gaussian,
}

impl SimpleKalman {
    pub fn new(x: Gaussian, fx: Gaussian) -> Self {
        Self { x, fx }
    }
    pub fn update(&mut self, z: Gaussian) -> Gaussian {
        // prediction
        let prediction = self.x + self.fx;

        // update
        let estimate = z * prediction;
        self.x = estimate;
        estimate
    }
}

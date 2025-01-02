#![allow(unused_imports)]
#![allow(non_snake_case)]
// #![no_std]

mod gh_filter;
use crate::gh_filter::GHFilter;
mod simple_kalman_filter;
use crate::simple_kalman_filter::{Gaussian, SimpleKalman};
mod multivariable_kalman_filter;
use crate::multivariable_kalman_filter::KalmanFilter;
use libm::powf;
use nalgebra::{Matrix2, Matrix2x1, SVector, Vector1, Vector2};

fn main() {
    let mut test_filter = GHFilter::new(100.0, 3.0, 0.3, 0.02);
    let x1 = test_filter.update(4.0, 0.1, None, None);

    println!("The prediction is {}, and the new gain is {}", x1.0, x1.1);

    let process_var = 1.0; // variance in the dog's movement
    let sensor_var = 2.0; // variance in the sensor

    let x = Gaussian::new(0.0, powf(20.0f32, 2.0)); //dog's position, N(0, 20**2)
    let velocity = 1.0;
    let dt = 1.0; // time step in seconds
    let process_model = Gaussian::new(velocity * dt, process_var); // displacement to add to x

    let mut dog = SimpleKalman::new(x, process_model);
    let measurement = Gaussian::new(1.354, sensor_var);
    let estimate = dog.update(measurement);
    println!(
        "The new estimate is {}, with a variance of {}",
        estimate.get_fields().0,
        estimate.get_fields().1
    );

    let x = Vector2::new(0.0, 0.0);
    let P = Matrix2::identity();
    let F = Matrix2::identity();
    let Q = Matrix2::identity() * 0.1;
    let B = Matrix2x1::new(1.0, 0.0); // Control input affects only first state
    let H = Matrix2::identity();

    let mut kf: KalmanFilter<f64, 2, 1> = KalmanFilter::new(x, P, F, Q, Some(B), H);

    let z = Vector2::new(1.0, 1.0);
    let R = Matrix2::identity() * 0.1;
    let u = Vector1::new(0.5); // Single control input

    kf.update(None, None, z, R, Some(u));

    let state = kf.get_state();
    println!("the controlled is {}", state[0]);
}

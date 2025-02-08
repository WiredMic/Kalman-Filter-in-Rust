#![allow(unused_imports)]
#![allow(non_snake_case)]
// #![no_std]

mod multivariable_kalman_filter;
use crate::multivariable_kalman_filter::KalmanFilter;
use core::f64;
use csv::Writer;
use libm::{cos, pow, sqrt};
use nalgebra::{Matrix1x3, Matrix3, Vector1, Vector3};
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::{error::Error, io, process};

// Make a Kalman filter that can track a 1D spring
//
// Math
// A spring works by pulling or pushing a mass
// The force the spring is pulled with is described by Hook's law
// \[F= -kx \]
// The force acting apon the spring can also be descriped by Newton's second law
// \[F=ma\]
// Equating these equations and solving for acceleration gives
// \[a=-\frac{k}{m}x\]
// Acceleration is the second derivetive of postion. This gives a differential equations.
// \[\frac{\mathrm{d}^2 x}{\mathrm{d}t^2}=-\frac{k}{m}x\]
// This differential equation has a solution
// \[x(t) = \cos \left( \sqrt{\frac{k}{m}}t + \phi  \right) \]
// With this equation an infinite amount of data can be generated
//
// Randomness
// This data needs some randomness.
// This randomness is modelled by a normal distrobution
// Get a point from a ND around 0.
// Then add the x postion from the equation above.

fn spring_model(t: f64) -> f64 {
    let k = 10.0; // newton per meter \(\frac{\mathrm{N}}{\mathrm{m}}  \)
    let m = 2.0; // kilogram \(\mathrm{kg}\)
    let phi = 0.0; // radians \(\mathrm{rad}\)

    cos(sqrt(k / m) * t + phi)
}

fn main() {
    let z_normal = Normal::new(0.0, 0.25).unwrap();
    let delta_time = 0.25; // second \(\mathrm{s}\)

    let mut data_points: Vec<[String; 5]> = vec![];

    // kalman filter
    let x = Vector3::new(1.0, 0.0, 0.0); // postion, velocity, acceleration
    let P = Matrix3::new(4.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0);
    // The state transition function is a description of how to predict the next prior
    // This Kalman filter will predict a point in motion. To do this Kinematic Equations will be used
    // \[x = x_0 + v_0 \Delta t + \frac{1}{2} a \Delta t ^2\]
    // \[v = v_0 + a \Delta t \]
    // \[a=a_0\]
    // The rest is 0
    // Because the difference in time never changes it can stay the same.
    let F = Matrix3::new(
        1.0,
        delta_time,
        0.5 * delta_time * delta_time,
        0.0,
        1.0,
        delta_time,
        0.0,
        0.0,
        1.0,
    );

    let Q = Matrix3::new(2.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0);

    let H = Matrix1x3::new(1.0, 0.0, 0.0);
    let mut kf: KalmanFilter<f64, 3, 1, 1> = KalmanFilter::new(x, P, F, Q, None, H);

    for i in 0..100 {
        let z_err = z_normal.sample(&mut rand::rng());
        let time = delta_time * (i as f64);
        let point = spring_model(time);

        // predict
        kf.predict(None, None, None);
        let prediction = kf.get_state().x;

        // update
        let z = Vector1::new(point + z_err);
        let R = Vector1::new(0.25);
        kf.update(z, R);
        let posterior = kf.get_state().x;

        // collect the different data
        data_points.push([
            time.to_string(),
            point.to_string(),
            prediction.to_string(),
            z.x.to_string(),
            posterior.to_string(),
        ]);
    }
    if let Err(err) = kalman_to_csv(data_points) {
        println!("error running example: {}", err);
        process::exit(1);
    }
}

fn kalman_to_csv(data_vec: Vec<[String; 5]>) -> Result<(), Box<dyn Error>> {
    let file = File::create("test.csv")?;
    let mut wtr = csv::Writer::from_writer(file);

    // When writing records without Serde, the header record is written just
    // like any other record.
    wtr.write_record(&["time", "x-point", "prediction", "measurement", "posterior"])?;
    for data in data_vec {
        wtr.write_record(&data)?;
    }

    wtr.flush()?;
    Ok(())
}

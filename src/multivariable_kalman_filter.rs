#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![warn(missing_docs)]

extern crate nalgebra as na;
// use core::fmt::Debug;
use na::base::{SMatrix, SVector};
use na::{ComplexField, RealField};

/// Kalman Filter
///
/// # Generic
/// `T` is the type of every element in the vectors and matrices
///
/// `X` is the size of the state estimate
///
/// `U` is the size of the column in the control input
///
/// `Z` is the size of the row in the measurement function
///
/// # Example
/// ```
/// let x = Vector2::new(0.0, 0.0);
/// let P = Matrix2::identity();
/// let F = Matrix2::identity();
/// let Q = Matrix2::identity() * 0.1;
/// let B = Matrix2x1::new(1.0, 0.0); // Control input affects only first state
/// let H = Matrix2::identity();
///
/// let mut kf: KalmanFilter<f64, 2, 2, 1> = KalmanFilter::new(x, P, F, Q, Some(B), H);
///
/// let z = Vector2::new(1.0, 1.0);
/// let R = Matrix2::identity() * 0.1;
/// let u = Vector1::new(0.5); // Single control input
///
/// kf.predict(None, None, Some(u));
/// kf.update(z, R);
///
/// let state = kf.get_state();
/// println!("the controlled is {}", state[0]);
/// ```
///
pub struct KalmanFilter<T: RealField + ComplexField, const X: usize, const Z: usize, const U: usize>
{
    /// The state estimate
    x: SVector<T, X>,
    /// The state covariance matrix
    P: SMatrix<T, X, X>,
    /// State transition matrix
    F: SMatrix<T, X, X>,
    /// Process noise covariance matrix
    Q: SMatrix<T, X, X>,
    /// Control input matrix (can handle U-dimensional input)
    B: Option<SMatrix<T, X, U>>,
    /// Measurement matrix
    H: SMatrix<T, Z, X>,
}

impl<T: RealField + ComplexField, const X: usize, const Z: usize, const U: usize>
    KalmanFilter<T, X, Z, U>
{
    /// # Input
    /// `x` is the initial state estiamate
    ///
    /// `P` is the initial state covariance matrix
    ///
    /// `F` is the state transition matrix
    ///
    /// `Q` is the process noice covariance matrix
    ///
    /// `B` is an optional control input matrix
    ///
    /// `H` is the measurement matrix
    ///
    pub fn new(
        x: SVector<T, X>,
        P: SMatrix<T, X, X>,
        F: SMatrix<T, X, X>,
        Q: SMatrix<T, X, X>,
        B: Option<SMatrix<T, X, U>>,
        H: SMatrix<T, Z, X>,
    ) -> Self {
        KalmanFilter { x, P, F, Q, B, H }
    }

    /// Reinitialize the state estimate and state covariance matrix
    ///
    /// This can be done to reuse the Kalman filter after a period of disuse
    pub fn reinitialize(&mut self, x: SVector<T, X>, P: SMatrix<T, X, X>) {
        // To be able to shutdown the filter and reuse the implimentation later
        self.x = x;
        self.P = P;
    }

    /// Set new control input matrix
    pub fn set_control_matrix(&mut self, B: SMatrix<T, X, U>) {
        self.B = Some(B);
    }

    /// # Prediction step of the Kalman filter
    /// ## input
    /// `F` change the state transition matrix
    ///
    /// `Q` change the process noice covariance matrix
    ///
    /// `u` is the control input vector
    pub fn predict(
        &mut self,
        F: Option<SMatrix<T, X, X>>,
        Q: Option<SMatrix<T, X, X>>,
        u: Option<SVector<T, U>>,
    ) {
        // Prediction step
        // Update state transition matrix if provided
        if let Some(new_F) = F {
            self.F = new_F;
        }

        // Update noise in system
        if let Some(Q) = Q {
            self.Q = Q;
        }

        // Calculate control input if provided
        let control_input = match (self.B.as_ref(), u.as_ref()) {
            (Some(B), Some(u)) => B * u,
            _ => SVector::<T, X>::zeros(),
        };

        // State prediction: \(\mathbf{F}\vec{x} + \mathbf{B}\vec{u} \)
        self.x = &self.F * &self.x + control_input;

        // Covariance prediction:  \(\mathbf{F}\mathbf{P}\mathbf{F}^\mathsf T + \mathbf{Q}  \)
        self.P = &self.F * &self.P * &self.F.transpose() + &self.Q;
    }

    /// Update step of the Kalman filter
    pub fn update(&mut self, z: SVector<T, Z>, R: SMatrix<T, Z, Z>) {
        // Update step
        // Measurement residual: \( \vec{z} - \mathbf{H}\bar{\vec{x}}\)
        let y = z - &self.H * &self.x;

        // Innovation covariance: \(\mathbf{S} = \mathbf{H\bar{P}H}^\mathsf T + \mathbf R\)
        let S = &self.H * &self.P * &self.H.transpose() + R;

        // Kalman gain: \( \mathbf K = \mathbf{\bar{P}H}^{\mathsf T} \mathbf{S}^{-1} \)
        let K = match S.try_inverse() {
            Some(S_inv) => &self.P * &self.H.transpose() * S_inv,
            None => {
                // Handle singular matrix case - could return Result instead
                return;
            }
        };

        // Update state estimate: \( \vec{x} = \bar{\vec{x}} + \mathbf{K}\vec{y}\)
        self.x = &self.x + &K * y;

        // Update covariance estimate: \(\mathbf P = (\mathbf I - \mathbf{KH})\mathbf{\bar{P}}  \)
        self.P = &self.P + K * &self.H * &self.P;
    }

    /// Get state estimate
    pub fn get_state(&self) -> &SVector<T, X> {
        &self.x
    }

    /// Get state covariance matrix
    pub fn get_covariance(&self) -> &SMatrix<T, X, X> {
        &self.P
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use na::{Matrix1, Matrix1x2, Matrix2, Matrix2x1, Vector1, Vector2};

    #[test]
    fn test_simple_kalman() {
        let x = Vector2::new(0.0, 0.0);
        let P = Matrix2::identity();
        let F = Matrix2::identity();
        let Q = Matrix2::identity() * 0.1;
        let H = Matrix2::identity();

        let mut kf: KalmanFilter<f32, 2, 2, 1> = KalmanFilter::new(x, P, F, Q, None, H);

        let z = Vector2::new(1.0, 1.0);
        let R = Matrix2::identity() * 0.1;

        kf.predict(None, None, None);
        kf.update(z, R);

        let state = kf.get_state();
        assert!((state[0] - 0.9).abs() < 0.1);
        assert!((state[1] - 0.9).abs() < 0.1);
    }

    #[test]
    fn test_with_control_input() {
        let x = Vector2::new(0.0, 0.0);
        let P = Matrix2::identity();
        let F = Matrix2::identity();
        let Q = Matrix2::identity() * 0.1;
        let B = Matrix2x1::new(1.0, 0.0); // Control input affects only first state
        let H = Matrix2::identity();

        let mut kf: KalmanFilter<f64, 2, 2, 1> = KalmanFilter::new(x, P, F, Q, Some(B), H);

        let z = Vector2::new(1.0, 1.0);
        let R = Matrix2::identity() * 0.1;
        let u = Vector1::new(0.5); // Single control input

        kf.predict(None, None, Some(u));
        kf.update(z, R);

        let state = kf.get_state();
        // First state should be affected by control input
        assert!((state[0] - 0.985).abs() < 0.1);
        // Second state should not be affected by control input
        assert!((state[1] - 0.9).abs() < 0.1);
    }
    #[test]
    fn test_with_different_measurement() {
        // State vector dimension = 2
        let x = Vector2::new(0.0, 0.0);
        let P = Matrix2::identity();
        let F = Matrix2::identity();
        let Q = Matrix2::identity() * 0.1;

        // Measurement matrix 1x2: maps 2D state to 1D measurement
        let H = Matrix1x2::new(1.0, 0.0); // Only measure first state component

        // Create Kalman filter with 2D state and 1D measurement
        let mut kf: KalmanFilter<f32, 2, 1, 1> = KalmanFilter::new(x, P, F, Q, None, H);

        // 1D measurement
        let z = Vector1::new(1.0);
        // 1x1 measurement noise covariance
        let R = Matrix1::new(0.1);

        kf.predict(None, None, None);
        kf.update(z, R);

        let state = kf.get_state();
        // First state should be updated based on measurement
        assert!((state[0] - 0.9).abs() < 0.1);
        // Second state should remain closer to initial value
        assert!((state[1]).abs() < 0.1);
    }
}

// struct that holds the estimate, the rate of change, size of timestep, g, and h of the filter
// an .update() function that takes a measument and gives the next estimate and rate of change
// It should also be able to take a new g and h value.
// start with a single dimensio

#[allow(dead_code)]
pub struct GHFilter {
    x: f32,  // unit
    dx: f32, // unit over time
    g: f32,
    h: f32,
}

impl GHFilter {
    #[allow(dead_code)]
    pub fn new(x: f32, dx: f32, g: f32, h: f32) -> Self {
        Self { x, dx, g, h }
    }
    #[allow(dead_code)]
    pub fn update(
        &mut self,
        z: f32,  // same unit as x
        dt: f32, // in seconds
        g: Option<f32>,
        h: Option<f32>,
    ) -> (f32, f32) {
        // update filter
        match g {
            Some(g) => self.g = g,
            None => {}
        }
        match h {
            Some(h) => self.h = h,
            None => {}
        }

        // prediction step
        let x_preduction = self.x + self.dx * dt;

        // update step
        let residual = z - x_preduction;
        self.dx += self.h * residual / dt;
        self.x = x_preduction + self.g * residual;

        (self.x, self.dx)
    }
}

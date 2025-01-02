#![no_std]

mod gh_filter;
mod multivariable_kalman_filter;
mod simple_kalman_filter;

// mod kalman_filter {
//     fn prediction_step(){

//     }

//     fn update_step(){

//     }
// }

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

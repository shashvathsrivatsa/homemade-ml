// use crate::*;


// ——— Hyperparameters ————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Hyperparameters {
    pub learning_rate: f64,
}

impl Hyperparameters {
    pub fn new() -> Self {
        Self {
            learning_rate: 0.05,
        }
    }
}


// use crate::*;

// ——— Hyperparameters ————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Hyperparameters {
    pub lr: f32,
    pub loss_threshold: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub dropout_rate: f32,
}

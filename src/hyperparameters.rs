use crate::*;

// ——— Hyperparameters ————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Hyperparameters {
    pub lr: f32,
    pub training_mode: TrainingMode,
    pub dropout_rate: f32,
    pub n_inputs: usize,
    pub n_layers: Vec<usize>,
    pub hidden_activation: Activation,
    pub output_activation: Activation,
    pub loss_function: LossFunction,
    pub optimizer: Optimizer,
}

pub enum TrainingMode {
    Full {
        loss_threshold: f32,
    },
    Batch {
        batch_size: usize,
        epochs: usize,
    }
}


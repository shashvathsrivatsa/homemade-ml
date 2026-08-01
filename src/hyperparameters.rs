use crate::*;

// ——— Hyperparameters ————————————————————————————————————————————————————————————————————————————————————————————————


// —— MLP ——————————————————————————————————————————————————————————————————————————————————
pub struct MlpHyperparameters {
    pub lr: f32,
    pub training_mode: TrainingMode,
    pub dropout_rate: f32,
    pub n_inputs: usize,
    pub n_layers: Vec<usize>,
    pub hidden_activation: Activation,
    pub output_activation: Activation,
    pub loss_function: LossFunction,
    pub optimizer: OptimizerSelector,
}

pub enum TrainingMode {
    Full {
        loss_threshold: f32,
    },
    Batch {
        batch_size: usize,
        epochs: usize,
    },
    OnePass
}

// —— DQN ——————————————————————————————————————————————————————————————————————————————————
pub struct DqnHyperparameters {
    pub model_hyperparameters: MlpHyperparameters,
    pub memory_capacity: usize,
    pub min_experiences: usize,
    pub decay: DecaySelector,
    pub eps_min: f32,
    pub batch_size: usize,
    pub gamma: f32,
    pub sync_freq: usize,
}


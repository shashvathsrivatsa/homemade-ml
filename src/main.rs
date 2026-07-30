use homemade_ml::*;

// ——— Main ———————————————————————————————————————————————————————————————————————————————————————————————————————————

fn main() {
    let mlp_hyperparameters = MlpHyperparameters {
        lr: 0.05,
        training_mode: TrainingMode::OnePass,
        dropout_rate: 0.0,
        n_inputs: 4,
        n_layers: vec![64, 2],
        hidden_activation: Relu,
        output_activation: NoActivation,
        loss_function: MSE,
        optimizer: AdamOptimizer,
    };

    let dqn_hyperparameters = DqnHyperparameters {
        model_hyperparameters: mlp_hyperparameters,
        memory_capacity: 10_000,
        min_experiences: 1_000,
        total_steps: 10_000,
        min_eps: 0.01,
        batch_size: 32,
        gamma: 0.95,
    };

    dqn(dqn_hyperparameters);
}


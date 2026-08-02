use homemade_ml::*;

// ——— Main ———————————————————————————————————————————————————————————————————————————————————————————————————————————

fn main() {
    let mlp_hyperparameters = MlpHyperparameters {
        lr: 0.003,
        training_mode: TrainingMode::OnePass,
        dropout_rate: 0.0,
        n_inputs: 13,
        n_layers: vec![64, 2],
        hidden_activation: Relu,
        output_activation: NoActivation,
        loss_function: MSE,
        optimizer: AdamOptimizer,
    };

    let dqn_hyperparameters = DqnHyperparameters {
        model_hyperparameters: mlp_hyperparameters,
        memory_capacity: 100_000,
        min_experiences: 10_000,
        decay: FlatDecay { min_eps: 0.1 },
        batch_size: 512,
        gamma: 0.99,
        sync_freq: 1000,
    };

    dqn_train(dqn_hyperparameters);
    // play_snake().unwrap();
}


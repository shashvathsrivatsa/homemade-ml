use homemade_ml::*;

// ——— Main ———————————————————————————————————————————————————————————————————————————————————————————————————————————

fn main() {
    let (xs, ys) = load_data(Train);

    let hyperparameters = Hyperparameters {
        lr: 0.05,
        training_mode: TrainingMode::Full { loss_threshold: 0.1 },
        dropout_rate: 0.2,
        n_inputs: 784,
        n_layers: vec![128, 64, 10],
        hidden_activation: Relu,
        output_activation: Softmax,
        loss_function: CrossEntropyLoss,
        optimizer: AdamOptimizer,
    };

    let mut model = MLP::new(hyperparameters);

    model.train(&xs, &ys);
    model.save();

    // model.load();
    // let accuracy = model.test(&xs, &ys);
    // println!("{:.2}%", accuracy * 100.0);
}


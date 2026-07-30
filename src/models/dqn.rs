use crate::*;

// ——— DQN ————————————————————————————————————————————————————————————————————————————————————————————————————————————

fn dqn() {
    let hyperparameters = Hyperparameters {
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

    let memory_capacity = 10_000;
    let min_experiences = 1_000;
    let total_steps = 10_000;
    let min_eps = 0.01;
    let batch_size = 32;

    // Warmup (populate memory)
    let mut state = State::new();
    let mut memory = Memory::new(memory_capacity, batch_size);

    for _ in 0..min_experiences {
        let action = random::<usize>() % 2;
        state.step(action, &mut memory);
    }

    // Episode loop
    let mut model = MLP::new(hyperparameters);
    let mut decay = LinearDecay::new(total_steps, min_eps);

    loop {

        // Pick epsilon-greedy action
        let explore = decay.explore();

        let action = if explore {
            random::<usize>() % 2
        } else {
            let y_pred = model.eval(&state.to_vec());
            y_pred.iter().enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap().0
        };

        // Step environment
        state.step(action, &mut memory);

        // Sample batch from memory
        let batch: Vec<&Experience> = memory.batch();

        // TODO: HOW DOES THIS MAKE ANY SENSE
        let (xs, ys): (Vec<&[f32]>, Vec<Vec<f32>>) = batch.iter().map(|exp| {
            let xs_i = exp.state.as_slice();

            let next_q = model.eval(&exp.next_state);
            let max_next_q = next_q.iter().cloned().reduce(f32::max).unwrap_or(0.0);
            let mut y_i = model.eval(&exp.state);

            y_i[exp.action as usize] = if exp.done {
                exp.reward
            } else {
                exp.reward + 0.99 * max_next_q
            };

            (xs_i, y_i)
        }).unzip();

        // Train model on batch
        // model.train(xs, ys);

        break;
    }
}


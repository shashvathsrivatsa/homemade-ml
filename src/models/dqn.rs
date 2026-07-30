use crate::*;

// ——— DQN ————————————————————————————————————————————————————————————————————————————————————————————————————————————

pub fn dqn(hyperparameters: DqnHyperparameters) {
    let mut model = MLP::new(hyperparameters.model_hyperparameters);
    let mut decay = LinearDecay::new(hyperparameters.total_steps, hyperparameters.min_eps);
    let mut graph = EpisodeGraph::new().unwrap();

    // Warmup (populate memory)
    let mut state = State::new();
    let mut memory = Memory::new(hyperparameters.memory_capacity, hyperparameters.batch_size);

    for _ in 0..hyperparameters.min_experiences {
        let action = random::<usize>() % 2;
        state.step(action, &mut memory, &mut graph);
    }

    // Episode loop
    loop {

        // Pick action (initially random, eventually purely best)
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
        state.step(action, &mut memory, &mut graph);

        // Sample batch from memory
        let batch: Vec<&Experience> = memory.batch();

        let (xs, ys): (Vec<Vec<f32>>, Vec<Vec<f32>>) = batch.iter().map(|exp| {
            let xs_i = exp.state.clone();

            let next_q = model.eval(&exp.next_state);
            let max_next_q = next_q.iter().cloned().reduce(f32::max).unwrap_or(0.0);
            let mut y_i = model.eval(&exp.state);

            y_i[exp.action as usize] = if exp.done {
                exp.reward
            } else {
                exp.reward + hyperparameters.gamma * max_next_q
            };

            (xs_i, y_i)
        }).unzip();

        // Train model on batch
        model.train(xs.as_slice(), ys.as_slice());
    }
}


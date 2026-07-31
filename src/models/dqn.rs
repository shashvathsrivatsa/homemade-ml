use crate::*;

// ——— DQN ————————————————————————————————————————————————————————————————————————————————————————————————————————————

pub fn dqn(hyperparameters: DqnHyperparameters) {
    let mut fast_model = MLP::new(&hyperparameters.model_hyperparameters);
    let mut slow_model = MLP::new(&hyperparameters.model_hyperparameters);
    let mut decay = LinearDecay::new(hyperparameters.total_steps, hyperparameters.min_eps);
    let mut graph = EpisodeGraph::new(10).unwrap();
    let mut state = State::new();
    let mut memory = Memory::new(hyperparameters.memory_capacity, hyperparameters.batch_size);
    fast_model.copy_weights_to(&mut slow_model);
    let mut best_episode_len = 100;

    // Warmup (populate memory)
    for _ in 0..hyperparameters.min_experiences {
        let action = random::<usize>() % 2;
        let step_result = state.step(action, &mut graph);
        if step_result.quit { return; }
        memory.push(step_result.experience);
    }

    // Episode loop
    let mut step = 0;
    loop {
        step += 1;

        // Pick action (initially random, eventually purely best)
        let explore = decay.explore();

        let action = if explore {
            random::<usize>() % 2
        } else {
            let y_pred = fast_model.eval(&state.to_vec());
            y_pred.iter().enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap().0
        };

        // Step environment
        let step_result = state.step(action, &mut graph);
        if step_result.quit { fast_model.save(false); return; };
        if step_result.experience.done && step_result.episode_len > best_episode_len {
            best_episode_len = step_result.episode_len;
            fast_model.save(true);
        }
        memory.push(step_result.experience);

        // Sample batch from memory
        let batch: Vec<&Experience> = memory.batch();

        let states: Vec<Vec<f32>> = batch.iter().map(|exp| exp.state.clone()).collect();
        let next_states: Vec<Vec<f32>> = batch.iter().map(|exp| exp.next_state.clone()).collect();

        let cur_q_vec = fast_model.eval_batch(&states);
        let next_q_vec = slow_model.eval_batch(&next_states);

        let (xs, ys): (Vec<Vec<f32>>, Vec<Vec<f32>>) = batch.iter().enumerate().map(|(i, exp)| {
            let xs_i = exp.state.clone();
            let mut y_i = cur_q_vec[i].clone();
            let next_q = next_q_vec[i].clone();
            let max_next_q = next_q.into_iter().reduce(f32::max).unwrap_or(0.0);

            y_i[exp.action as usize] = if exp.done {
                exp.reward
            } else {
                exp.reward + hyperparameters.gamma * max_next_q
            };

            (xs_i, y_i)
        }).unzip();

        // Train model on batch
        fast_model.train(xs.as_slice(), ys.as_slice());

        // Periodically sync networks
        if step % hyperparameters.sync_freq == 0 { fast_model.copy_weights_to(&mut slow_model); }
    }
}

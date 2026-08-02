use crate::*;

// ——— DQN ————————————————————————————————————————————————————————————————————————————————————————————————————————————

pub fn dqn_train(hyperparameters: DqnHyperparameters) {
    let mut fast_model = MLP::new(&hyperparameters.model_hyperparameters);
    let mut slow_model = MLP::new(&hyperparameters.model_hyperparameters);
    let mut decay: Decay = hyperparameters.decay.into_decay();
    // let mut graph = FruitsGraph::new(10).unwrap();
    let mut graph = SnakeVisualization::new(60).unwrap();
    let mut state = State::new();
    let mut memory = Memory::new(hyperparameters.memory_capacity, hyperparameters.batch_size);
    let mut best_fruits_eaten = 0;

    // fast_model.load(true);
    fast_model.copy_weights_to(&mut slow_model);

    // Warmup (populate memory)
    for _ in 0..hyperparameters.min_experiences {
        let action = random::<usize>() % 3;
        let step_result = state.step(action, None);
        if matches!(step_result.key_pressed, Some('q' | '\u{3}')) { return; }
        memory.push(step_result.experience);
    }

    // Episode loop
    let mut step = 0;
    loop {
        step += 1;

        let action = if decay.explore() {
            random::<usize>() % 3
        } else {
            let y_pred = fast_model.eval(&state.to_vec());
            y_pred.iter().enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap().0
        };

        // Step environment
        let step_result = state.step(action, Some(&mut graph));

        if let Some(key) = step_result.key_pressed {
            match key {
                'q' | '\u{3}' => return,
                's' => { fast_model.save(true); return; }
                _ => ()
            }
        };

        if step_result.experience.done && step_result.fruits_eaten > best_fruits_eaten {
            best_fruits_eaten = step_result.fruits_eaten;
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

pub fn test_dqn(hyperparameters: DqnHyperparameters) {
    let mut model = MLP::new(&hyperparameters.model_hyperparameters);
    let mut state = State::new();
    let mut visualization = FruitsGraph::new(10).unwrap();
    model.load(true);

    // Episode loop
    loop {

        // Pick model-predicted action
        let y_pred = model.eval(&state.to_vec());
        let action = y_pred.iter().enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap().0;

        // Step environment
        let step_result = state.step(action, Some(&mut visualization));
        // std::thread::sleep(Duration::from_millis(100));
        if matches!(step_result.key_pressed, Some('q' | '\u{3}')) || step_result.experience.done { return; }
    }
}

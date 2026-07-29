use crate::*;

// ——— DQN ————————————————————————————————————————————————————————————————————————————————————————————————————————————

fn dqn(hyperparameters: Hyperparameters) {

    // Warmup (populate memory)
    let mut state = State::new();
    let memory = Memory::new();

    // Episode loop
    let model = MLP::new(hyperparameters);

    loop {
        let mut state = State::new();


        // Step loop
        loop {
            // let done = state.step();

            break;
        }

        break;
    }
}


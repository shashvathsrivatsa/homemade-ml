use crate::*;

// ——— DQN ————————————————————————————————————————————————————————————————————————————————————————————————————————————

fn dqn(hyperparameters: Hyperparameters) {
    let min_experiences = 1_000;
    let total_steps = 10_000;

    // Warmup (populate memory)
    let mut state = State::new();
    let mut memory = Memory::new();

    for _ in 0..min_experiences {
        let action = random::<usize>() % 2;
        state.step(action, &mut memory);
    }

    // Episode loop
    let model = MLP::new(hyperparameters);
    let mut state = State::new();
    let linear_decay = LinearDecay::new(total_steps);

    loop {
        let eps = random::<f32>();

        break;
    }
}


pub struct LinearDecay {
    total_steps: usize,
    step: usize,
}

impl LinearDecay {
    pub fn new(total_steps: usize) -> Self {
        Self { total_steps, step: 0 }
    }

    pub fn 
}


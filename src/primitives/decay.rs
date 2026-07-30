use crate::*;

// ——— Loss ———————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct LinearDecay {
    total_steps: f32,
    min_eps: f32,
    step: f32,
}

impl LinearDecay {
    pub fn new(total_steps: usize, min_eps: f32) -> Self {
        Self { total_steps: total_steps as f32, min_eps, step: 0.0 }
    }

    // true = explore, false = exploit
    pub fn explore(&mut self) -> bool {
        self.step += 1.0;
        let eps = (1.0 - self.step / self.total_steps).max(self.min_eps);
        return random::<f32>() < eps;
    }
}



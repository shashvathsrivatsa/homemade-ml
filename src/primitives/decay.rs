use crate::*;

// ——— Decay ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub enum DecaySelector {
    LinearDecay { total_steps: usize, min_eps: f32 }
}

impl DecaySelector {
    pub fn into_decay(&self) -> Decay {
        match self {
            DecaySelector::LinearDecay { total_steps, min_eps } => Decay::LinearDecay(LinearDecay::new(*total_steps, *min_eps)),
        }
    }
}

pub enum Decay {
    LinearDecay(LinearDecay),
}

impl Decay {
    pub fn explore(&mut self) -> bool {
        match self {
            Decay::LinearDecay(d) => d.explore(),
        }
    }
}


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



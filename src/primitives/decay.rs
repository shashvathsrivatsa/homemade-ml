use crate::*;

// ——— Decay ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub enum DecaySelector {
    LinearDecay { total_steps: usize, min_eps: f32 },
    FlatDecay { min_eps: f32 },
}

pub use DecaySelector::*;

impl DecaySelector {
    pub fn into_decay(&self) -> Decay {
        match self {
            DecaySelector::LinearDecay { total_steps, min_eps } => Decay::LinearDecay(LinearDecayState::new(*total_steps, *min_eps)),
            DecaySelector::FlatDecay { min_eps } => Decay::FlatDecay(FlatDecayState::new(*min_eps)),
        }
    }
}

pub enum Decay {
    LinearDecay(LinearDecayState),
    FlatDecay(FlatDecayState),
}

impl Decay {
    // true = explore, false = exploit
    pub fn explore(&mut self) -> bool {
        match self {
            Decay::LinearDecay(d) => d.explore(),
            Decay::FlatDecay(d) => d.explore(),
        }
    }
}


// —— Linear Decay —————————————————————————————————————————————————————————————————————————

pub struct LinearDecayState {
    total_steps: f32,
    min_eps: f32,
    step: f32,
}

impl LinearDecayState {
    pub fn new(total_steps: usize, min_eps: f32) -> Self {
        Self { total_steps: total_steps as f32, min_eps, step: 0.0 }
    }

    pub fn explore(&mut self) -> bool {
        self.step += 1.0;
        let eps = (1.0 - self.step / self.total_steps).max(self.min_eps);
        return random::<f32>() < eps;
    }
}


// —— Flat Decay —————————————————————————————————————————————————————————————————————————

pub struct FlatDecayState {
    min_eps: f32,
}

impl FlatDecayState {
    pub fn new(min_eps: f32) -> Self {
        Self { min_eps }
    }

    pub fn explore(&mut self) -> bool {
        return random::<f32>() < self.min_eps;
    }
}


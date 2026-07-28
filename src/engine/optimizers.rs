use crate::*;

// ——— Loss ———————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct AdamOptimizer {
    b1: f32,
    b2: f32,
    m: f32,
    v: f32,
    t: i32,
}

impl AdamOptimizer {
    pub fn new() -> Self {
        Self {
            b1: 0.9,
            b2: 0.999,
            m: 0.0,
            v: 0.0,
            t: 0,
        }
    }

    pub fn update(&mut self, lr: f32, grad: f32, param: f32) -> f32 {
        self.t += 1;
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad;
        self.v = self.b2 * self.v + (1.0 - self.b2) * grad.powi(2);
        let m_hat = self.m / (1.0 - self.b1.powi(self.t));
        let v_hat = self.v / (1.0 - self.b2.powi(self.t));
        let dp = lr * m_hat / (v_hat + 1e-8).sqrt();
        param - dp
    }
}


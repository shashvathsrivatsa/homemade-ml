use crate::*;

// ——— Activation —————————————————————————————————————————————————————————————————————————————————————————————————————

pub enum Activation {
    Tanh,
    Relu,
    Softmax,
    NoActivation,
}

pub use Activation::*;

impl Activation {
    pub fn apply(&self, pool: &mut Pool, logits: Tensor) -> Tensor {
        match self {
            Tanh => pool.tanh(logits),
            Relu => relu(pool, logits),
            Softmax => softmax(pool, logits),
            NoActivation => logits,
        }
    }
}

pub fn relu(pool: &mut Pool, logits: Tensor) -> Tensor {
    let zeros = pool.fill(pool.get_shape(logits).to_vec(), 0.0);
    pool.max(logits, zeros)
}

pub fn softmax(pool: &mut Pool, logits: Tensor) -> Tensor {
    let safe_logits = pool.sub_row_max(logits);
    let exps = pool.exp(safe_logits);
    let sum = pool.sum(exps);
    pool.div(exps, sum)
}

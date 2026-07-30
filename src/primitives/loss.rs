use crate::*;

// ——— Loss ———————————————————————————————————————————————————————————————————————————————————————————————————————————

pub enum LossFunction {
    MSE,
    CrossEntropyLoss,
}

pub use LossFunction::*;

impl LossFunction {
    pub fn apply(&self, pool: &mut Pool, y_pred: Tensor, labels: Tensor) -> Tensor {
        match self {
            MSE => mse(pool, y_pred, labels),
            CrossEntropyLoss => cross_entropy_loss(pool, y_pred, labels),
        }
    }
}

// [batches, labels] x [batches, labels] => loss for each batch [batches]  => avg loss [1]
pub fn mse(pool: &mut Pool, y_pred: Tensor, labels: Tensor) -> Tensor {
    let sub = pool.sub(y_pred, labels);
    let sq = pool.sq(sub);
    let mean_row = pool.mean(sq);
    let loss = pool.mean(mean_row);
    loss
}

pub fn cross_entropy_loss(pool: &mut Pool, y_pred: Tensor, labels: Tensor) -> Tensor {
    let y_prob = pool.gather(y_pred, labels);
    let eps = pool.fill(pool.get_shape(y_prob).to_vec(), 1e-7);
    let y_prob_safe = pool.max(y_prob, eps);
    let log = pool.log(y_prob_safe);
    let mean = pool.mean(log);
    let neg = pool.neg(mean);
    neg
}

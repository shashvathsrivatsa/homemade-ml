use crate::*;


// —— Activation ———————————————————————————————————————————————————————————————————————————

pub enum Activation {
    Tanh,
    Softmax,
}

pub use Activation::*;

impl Activation {
    pub fn apply(&self, pool: &mut Pool, logits: &[Value]) -> Vec<Value> {
        match self {
            Activation::Tanh => logits.iter().map(|&y| tanh(pool, y)).collect(),
            Activation::Softmax => softmax(pool, logits),
        }
    }
}

pub fn tanh(pool: &mut Pool, logit: Value) -> Value {
    let two = pool.new_value(2.0);
    let one = pool.new_value(1.0);
    let twox = pool.mul(two, logit);
    let exp = pool.exp(twox);
    let sub = pool.sub(exp, one);
    let add = pool.add(exp, one);
    pool.div(sub, add)
}

pub fn relu(pool: &mut Pool, logit: Value) -> Value {

    Value(0)
}

pub fn softmax(pool: &mut Pool, logits: &[Value]) -> Vec<Value> {
    let exps: Vec<Value> = logits.iter().map(|&yout| pool.exp(yout)).collect();
    let sum = exps.iter().fold(pool.new_value(0.0), |acc, &e| pool.add(acc, e));
    exps.iter().map(|&e| pool.div(e, sum)).collect()
}

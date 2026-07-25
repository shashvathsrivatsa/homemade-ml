use crate::*;


// —— Loss —————————————————————————————————————————————————————————————————————————————

// pub fn mse_loss(pool: &mut Pool, ys: &[Value], y_pred: &[Value]) -> Value {
//     ys.iter().zip(y_pred.iter()).fold(pool.new_value(0.0), |acc, (&ygt, &yout)| {
//         let diff = pool.sub(ygt, yout);
//         let sq = pool.pow2(diff);
//         pool.add(acc, sq)
//     })
// }

pub fn cross_entropy_loss(pool: &mut TensorPool, y_pred: Tensor, labels: Tensor) -> Tensor {
    let y_prob = pool.gather(y_pred, labels);
    let log = pool.log(y_prob);
    let mean = pool.mean(log);
    let neg = pool.neg(mean);
    neg
}


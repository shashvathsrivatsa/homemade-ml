use crate::*;


// —— Loss —————————————————————————————————————————————————————————————————————————————

pub fn mse_loss(pool: &mut Pool, ys: &[Value], y_pred: &[Value]) -> Value {
    ys.iter().zip(y_pred.iter()).fold(pool.new_value(0.0), |acc, (&ygt, &yout)| {
        let diff = pool.sub(ygt, yout);
        let sq = pool.pow2(diff);
        pool.add(acc, sq)
    })
}


use crate::*;


// —— Loss —————————————————————————————————————————————————————————————————————————————

pub fn mse_loss(pool: &mut Pool, ys: &[Value], y_pred: &[Value]) -> Value {
    ys.iter().zip(y_pred.iter()).fold(pool.new_value(0.0), |acc, (&ygt, &yout)| {
        let diff = pool.sub(ygt, yout);
        let sq = pool.pow2(diff);
        pool.add(acc, sq)
    })
}

pub fn cross_entropy_loss(pool: &mut Pool, ys: &[Value], y_pred: &[Vec<Value>]) -> Value {
    let sum = ys.iter().zip(y_pred.iter()).fold(pool.new_value(0.0), |acc, (ygt, yout)| {
        let label: usize = pool.get_data(*ygt) as usize;
        let log = pool.log(yout[label]);
        pool.add(acc, log)
    });

    let neg = pool.neg(sum);
    let inv_n = pool.new_value(1.0 / ys.len() as f64);
    pool.mul(neg, inv_n)
}


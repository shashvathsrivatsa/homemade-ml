use micrograd::*;


// ——— Main ———————————————————————————————————————————————————————————————————————————————————————————————————————————

fn main() {

    let xs: Vec<Vec<Value>> = vec![
        vec![2.0,  3.0, -1.0],
        vec![3.0, -1.0,  0.5],
        vec![0.5,  1.0,  1.0],
        vec![1.0,  1.0, -1.0],
    ].iter().map(|row| {
        row.iter().map(|&entry| Value::new(entry)).collect()
    }).collect();

    let ys: Vec<Value> = vec![1.0, -1.0, -1.0, 1.0].iter().map(|&entry| Value::new( entry)).collect();

    let n = MLP::new(3, vec![4, 4, 1]);

    let y_pred: Vec<Value> = xs.iter().map(|x| n.call(&x)[0].clone()).collect();
    y_pred.iter().for_each(|y_pred| println!("{:?}", y_pred));

    let loss = ys.iter().zip(y_pred.iter()).fold(
        Value::new(0.0), |acc, (ygt, yout)| acc + (ygt - yout).powi(2)
    );

    println!("Loss: {:?}", loss);
    draw_dot(&loss);
}


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

    let mut mlp = MLP::new(3, vec![4, 4, 1]);

    mlp.train(&xs, &ys, &Hyperparameters::new());

    println!("{:?}", mlp.eval(&xs[0]));
}


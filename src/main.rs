use homemade_ml::*;

// ——— Main ———————————————————————————————————————————————————————————————————————————————————————————————————————————

fn main() {
    let hyperparameters = Hyperparameters {
        learning_rate: 0.05,
        loss_threshold: 0.05,
        batch_size: 20000,
        epochs: 10,
    };
    let mut mlp = MLP::new(784, vec![128, 64, 10], Relu, Softmax, hyperparameters);

    train(&mut mlp);
}

#[allow(dead_code)]
fn train(mlp: &mut MLP) {
    println!("Loading data...");
    let (xs, ys) = load_data(Train);

    println!("Training...");
    mlp.train_full(&xs, &ys);

    mlp.save();
}

#[allow(dead_code)]
fn test(mlp: &mut MLP) {
    mlp.load();

    println!("Loading data...");
    let (xs, ys) = load_data(Test);

    println!("Testing...");
    let accuracy = mlp.test(&xs, &ys);
    println!("{:.2}%", accuracy * 100.0);
}

#[allow(dead_code)]
fn live_test(mlp: &mut MLP) {
    mlp.load();
    let x: Vec<f32> = load_bw_png("data/live.png".to_string());

    let y = mlp.eval(&x);

    print!("\n  ");
    (0..=9).for_each(|i| print!("{}    ", i));
    println!();
    y.iter().for_each(|n| print!("{:.2} ", n));
    println!("\n");

    println!(
        "{}\n",
        (0..=9).fold(0, |max_i, i| if y[i] > y[max_i] { i } else { max_i })
    );
}


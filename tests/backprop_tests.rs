use homemade_ml::*;

fn assert_gradient(pool: &mut Pool, tensor: Tensor, before: &[f32], expected: &[f32]) {
    pool.update(tensor, 1.0);

    for ((before, after), expected) in before
        .iter()
        .zip(pool.get_data(tensor))
        .zip(expected.iter())
    {
        let actual = before - after;
        assert!(
            (actual - expected).abs() < 1e-5,
            "gradient mismatch: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn mul_backward() {
    let mut pool = Pool::new();
    let a_data = vec![2.0, 3.0];
    let b_data = vec![4.0, 5.0];
    let a = pool.new_tensor(a_data.clone(), vec![2]);
    let b = pool.new_tensor(b_data.clone(), vec![2]);
    let out = pool.mul(a, b);

    pool.backpropogate(out);

    assert_gradient(&mut pool, a, &a_data, &[4.0, 5.0]);
    assert_gradient(&mut pool, b, &b_data, &[2.0, 3.0]);
}

#[test]
fn tanh_backward() {
    let mut pool = Pool::new();
    let input_data = vec![0.5, -0.25];
    let input = pool.new_tensor(input_data.clone(), vec![2]);
    let out = pool.tanh(input);
    let expected: Vec<f32> = pool
        .get_data(out)
        .iter()
        .map(|value| 1.0 - value.powi(2))
        .collect();

    pool.backpropogate(out);

    assert_gradient(&mut pool, input, &input_data, &expected);
}

#[test]
fn chain_mul_tanh_backward() {
    let mut pool = Pool::new();
    let a_data = vec![0.5, -1.0];
    let b_data = vec![2.0, 0.25];
    let a = pool.new_tensor(a_data.clone(), vec![2]);
    let b = pool.new_tensor(b_data.clone(), vec![2]);
    let product = pool.mul(a, b);
    let out = pool.tanh(product);
    let tanh_grad: Vec<f32> = pool
        .get_data(out)
        .iter()
        .map(|value| 1.0 - value.powi(2))
        .collect();
    let expected_a: Vec<f32> = tanh_grad
        .iter()
        .zip(&b_data)
        .map(|(grad, b)| grad * b)
        .collect();
    let expected_b: Vec<f32> = tanh_grad
        .iter()
        .zip(&a_data)
        .map(|(grad, a)| grad * a)
        .collect();

    pool.backpropogate(out);

    assert_gradient(&mut pool, a, &a_data, &expected_a);
    assert_gradient(&mut pool, b, &b_data, &expected_b);
}

#[test]
fn shared_parent_accumulates_gradients() {
    let mut pool = Pool::new();
    let input_data = vec![2.0, -3.0];
    let input = pool.new_tensor(input_data.clone(), vec![2]);
    let squared = pool.mul(input, input);

    pool.backpropogate(squared);

    assert_gradient(&mut pool, input, &input_data, &[4.0, -6.0]);
}

#[test]
fn matmul_backward() {
    let mut pool = Pool::new();
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];
    let a = pool.new_tensor(a_data.clone(), vec![2, 2]);
    let b = pool.new_tensor(b_data.clone(), vec![2, 2]);
    let out = pool.matmul(a, b);

    pool.backpropogate(out);

    assert_gradient(&mut pool, a, &a_data, &[11.0, 15.0, 11.0, 15.0]);
    assert_gradient(&mut pool, b, &b_data, &[4.0, 4.0, 6.0, 6.0]);
}

#[test]
fn bias_add_backward() {
    let mut pool = Pool::new();
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let bias_data = vec![0.5, -0.5];
    let input = pool.new_tensor(input_data.clone(), vec![2, 2]);
    let bias = pool.new_tensor(bias_data.clone(), vec![2]);
    let out = pool.bias_add(input, bias);

    pool.backpropogate(out);

    assert_gradient(&mut pool, input, &input_data, &[1.0; 4]);
    assert_gradient(&mut pool, bias, &bias_data, &[2.0, 2.0]);
}

#[test]
fn cross_entropy_backward() {
    let mut pool = Pool::new();
    let predictions_data = vec![0.2, 0.5, 0.3, 0.1, 0.3, 0.6];
    let labels_data = vec![1.0, 2.0];
    let predictions = pool.new_tensor(predictions_data.clone(), vec![2, 3]);
    let labels = pool.new_tensor(labels_data.clone(), vec![2]);
    let loss = cross_entropy_loss(&mut pool, predictions, labels);

    assert!((pool.get_data(loss)[0] + (0.5_f32.ln() + 0.6_f32.ln()) / 2.0).abs() < 1e-5);

    pool.backpropogate(loss);

    assert_gradient(
        &mut pool,
        predictions,
        &predictions_data,
        &[0.0, -1.0, 0.0, 0.0, 0.0, -1.0 / 1.2],
    );
    assert_gradient(&mut pool, labels, &labels_data, &[0.0, 0.0]);
}

#[test]
fn softmax_rows_sum_to_one_and_backward_is_zero_for_total() {
    let mut pool = Pool::new();
    let logits_data = vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0];
    let logits = pool.new_tensor(logits_data.clone(), vec![2, 3]);
    let probabilities = softmax(&mut pool, logits);

    for row in pool.get_data(probabilities).chunks(3) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    let row_sums = pool.sum(probabilities);
    let total = pool.mean(row_sums);
    pool.backpropogate(total);

    assert_gradient(&mut pool, logits, &logits_data, &[0.0; 6]);
}

#[test]
fn softmax_cross_entropy_logits_gradient_matches_analytical_result() {
    let mut pool = Pool::new();
    let logits_data = vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0];
    let logits = pool.new_tensor(logits_data, vec![2, 3]);
    let probabilities = softmax(&mut pool, logits);
    let probability_data = pool.get_data(probabilities).to_vec();
    let labels = pool.new_tensor(vec![2.0, 0.0], vec![2]);
    let loss = cross_entropy_loss(&mut pool, probabilities, labels);

    pool.backpropogate(loss);

    let mut expected = probability_data;
    expected[2] -= 1.0;
    expected[3] -= 1.0;
    expected.iter_mut().for_each(|gradient| *gradient /= 2.0);

    eprintln!("actual logits gradient:   {:?}", pool.get_grad(logits));
    eprintln!("expected logits gradient: {expected:?}");

    for (actual, expected) in pool.get_grad(logits).iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "softmax-cross-entropy logits gradient mismatch: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn relu_backward() {
    let mut pool = Pool::new();
    let input_data = vec![-2.0, 0.0, 3.0];
    let input = pool.new_tensor(input_data.clone(), vec![1, 3]);
    let out = relu(&mut pool, input);

    assert_eq!(pool.get_data(out), &[0.0, 0.0, 3.0]);

    pool.backpropogate(out);

    assert_gradient(&mut pool, input, &input_data, &[0.0, 0.0, 1.0]);
}

#[test]
fn mlp_eval_runs_after_pool_rename() {
    let hyperparameters = Hyperparameters {
        learning_rate: 0.05,
        loss_threshold: 0.0,
        batch_size: 32,
        epochs: 10,
    };
    let mut mlp = MLP::new(3, vec![4, 2], Tanh, Softmax, hyperparameters);

    let first = mlp.eval(&[2.0, 3.0, 4.0]);
    let second = mlp.eval(&[2.0, 3.0, 4.0]);

    assert_eq!(first.len(), 2);
    assert!(first.iter().all(|value| value.is_finite()));
    assert!((first.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert_eq!(second.len(), 2);
}

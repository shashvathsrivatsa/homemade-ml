use micrograd::*;

fn numerical_grad(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-5;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

#[test]
fn test_add_backward() {
    let mut pool = Pool::new();
    let a = pool.new_value(2.0);
    let b = pool.new_value(3.0);
    let c = pool.add(a, b);

    pool.backpropogate(c);

    assert_eq!(pool.get_grad(a), 1.0);
    assert_eq!(pool.get_grad(b), 1.0);
}

#[test]
fn test_mul_backward() {
    let mut pool = Pool::new();
    let a = pool.new_value(2.0);
    let b = pool.new_value(3.0);
    let c = pool.mul(a, b);

    pool.backpropogate(c);

    assert_eq!(pool.get_grad(a), 3.0);
    assert_eq!(pool.get_grad(b), 2.0);
}

#[test]
fn test_tanh_backward() {
    let mut pool = Pool::new();
    let a = pool.new_value(0.5);
    let out = tanh(&mut pool, a);

    pool.backpropogate(out);

    let value = pool.get_data(out);
    let expected = 1.0 - value * value;
    assert!((pool.get_grad(a) - expected).abs() < 1e-9);
}

#[test]
fn test_chain_mul_tanh() {
    let mut pool = Pool::new();
    let a = pool.new_value(0.5);
    let b = pool.new_value(2.0);
    let product = pool.mul(a, b);
    let out = tanh(&mut pool, product);

    pool.backpropogate(out);

    let value = pool.get_data(out);
    let dtanh = 1.0 - value * value;
    assert!((pool.get_grad(a) - dtanh * 2.0).abs() < 1e-9);
    assert!((pool.get_grad(b) - dtanh * 0.5).abs() < 1e-9);
}

#[test]
fn test_leaf_diamond() {
    let mut pool = Pool::new();
    let a = pool.new_value(2.0);
    let b = pool.new_value(3.0);
    let c = pool.new_value(4.0);
    let ab = pool.mul(a, b);
    let ac = pool.mul(a, c);
    let out = pool.add(ab, ac);

    pool.backpropogate(out);

    assert_eq!(pool.get_grad(a), 7.0);
    assert_eq!(pool.get_grad(b), 2.0);
    assert_eq!(pool.get_grad(c), 2.0);
}

#[test]
fn test_nonleaf_diamond() {
    let mut pool = Pool::new();
    let a = pool.new_value(2.0);
    let b = pool.new_value(3.0);
    let c = pool.new_value(4.0);
    let d = pool.new_value(5.0);
    let shared = pool.add(a, b);
    let left = pool.mul(shared, c);
    let right = pool.mul(shared, d);
    let out = pool.add(left, right);

    pool.backpropogate(out);

    assert_eq!(pool.get_grad(c), 5.0);
    assert_eq!(pool.get_grad(d), 5.0);
    assert_eq!(pool.get_grad(a), 9.0);
    assert_eq!(pool.get_grad(b), 9.0);
}

fn build_1in_2h_1out(
    xv: f64,
    w00v: f64,
    b0v: f64,
    w10v: f64,
    b1v: f64,
    w20v: f64,
    w21v: f64,
    b2v: f64,
) -> (Pool, Value, [Value; 5]) {
    let mut pool = Pool::new();
    let x = pool.new_value(xv);
    let w00 = pool.new_value(w00v);
    let b0 = pool.new_value(b0v);
    let w10 = pool.new_value(w10v);
    let b1 = pool.new_value(b1v);
    let w20 = pool.new_value(w20v);
    let w21 = pool.new_value(w21v);
    let b2 = pool.new_value(b2v);

    let xw00 = pool.mul(x, w00);
    let h0_sum = pool.add(xw00, b0);
    let h0 = tanh(&mut pool, h0_sum);
    let xw10 = pool.mul(x, w10);
    let h1_sum = pool.add(xw10, b1);
    let h1 = tanh(&mut pool, h1_sum);
    let h0w20 = pool.mul(h0, w20);
    let h1w21 = pool.mul(h1, w21);
    let out_sum = pool.add(h0w20, h1w21);
    let out_sum = pool.add(out_sum, b2);
    let out = tanh(&mut pool, out_sum);

    (pool, out, [x, w00, w10, w20, w21])
}

fn eval_1in_2h_1out(
    xv: f64,
    w00: f64,
    b0: f64,
    w10: f64,
    b1: f64,
    w20: f64,
    w21: f64,
    b2: f64,
) -> f64 {
    let (pool, out, _) = build_1in_2h_1out(xv, w00, b0, w10, b1, w20, w21, b2);
    pool.get_data(out)
}

#[test]
fn test_mlp_single_output_neuron_no_sharing() {
    let (xv, w00, b0, w10, b1, w20, w21, b2) =
        (0.5, 0.3, 0.1, -0.4, 0.2, 0.6, -0.2, 0.05);
    let (mut pool, out, [x, w00_handle, w10_handle, w20_handle, w21_handle]) =
        build_1in_2h_1out(xv, w00, b0, w10, b1, w20, w21, b2);

    pool.backpropogate(out);

    let cases = [
        (x, pool.get_grad(x), numerical_grad(|v| eval_1in_2h_1out(v, w00, b0, w10, b1, w20, w21, b2), xv)),
        (w00_handle, pool.get_grad(w00_handle), numerical_grad(|v| eval_1in_2h_1out(xv, v, b0, w10, b1, w20, w21, b2), w00)),
        (w10_handle, pool.get_grad(w10_handle), numerical_grad(|v| eval_1in_2h_1out(xv, w00, b0, v, b1, w20, w21, b2), w10)),
        (w20_handle, pool.get_grad(w20_handle), numerical_grad(|v| eval_1in_2h_1out(xv, w00, b0, w10, b1, v, w21, b2), w20)),
        (w21_handle, pool.get_grad(w21_handle), numerical_grad(|v| eval_1in_2h_1out(xv, w00, b0, w10, b1, w20, v, b2), w21)),
    ];

    for (handle, analytic, numeric) in cases {
        assert!((analytic - numeric).abs() < 1e-4, "gradient mismatch for value {}: analytic={analytic}, numeric={numeric}", handle.0);
    }
}

fn fwd_2layer2_neurons(
    xv: f64,
    w00v: f64,
    b0v: f64,
    w10v: f64,
    b1v: f64,
    w20v: f64,
    w21v: f64,
    b2v: f64,
    w30v: f64,
    w31v: f64,
    b3v: f64,
) -> f64 {
    let mut pool = Pool::new();
    let x = pool.new_value(xv);
    let w00 = pool.new_value(w00v);
    let b0 = pool.new_value(b0v);
    let w10 = pool.new_value(w10v);
    let b1 = pool.new_value(b1v);
    let w20 = pool.new_value(w20v);
    let w21 = pool.new_value(w21v);
    let b2 = pool.new_value(b2v);
    let w30 = pool.new_value(w30v);
    let w31 = pool.new_value(w31v);
    let b3 = pool.new_value(b3v);

    let xw00 = pool.mul(x, w00);
    let h0_sum = pool.add(xw00, b0);
    let h0 = tanh(&mut pool, h0_sum);
    let xw10 = pool.mul(x, w10);
    let h1_sum = pool.add(xw10, b1);
    let h1 = tanh(&mut pool, h1_sum);

    let h0w20 = pool.mul(h0, w20);
    let h1w21 = pool.mul(h1, w21);
    let n20_sum = pool.add(h0w20, h1w21);
    let n20_sum = pool.add(n20_sum, b2);
    let n20 = tanh(&mut pool, n20_sum);

    let h0w30 = pool.mul(h0, w30);
    let h1w31 = pool.mul(h1, w31);
    let n21_sum = pool.add(h0w30, h1w31);
    let n21_sum = pool.add(n21_sum, b3);
    let n21 = tanh(&mut pool, n21_sum);
    let out = pool.add(n20, n21);

    pool.get_data(out)
}

#[test]
fn test_mlp_two_layer2_neurons_numeric_grad() {
    let values = (0.5, 0.3, 0.1, -0.4, 0.2, 0.6, -0.2, 0.05, -0.5, 0.7, -0.1);
    let (xv, w00, b0, w10, b1, w20, w21, b2, w30, w31, b3) = values;

    let mut pool = Pool::new();
    let x = pool.new_value(xv);
    let w00_handle = pool.new_value(w00);
    let b0_handle = pool.new_value(b0);
    let w10_handle = pool.new_value(w10);
    let b1_handle = pool.new_value(b1);
    let w20_handle = pool.new_value(w20);
    let w21_handle = pool.new_value(w21);
    let b2_handle = pool.new_value(b2);
    let w30_handle = pool.new_value(w30);
    let w31_handle = pool.new_value(w31);
    let b3_handle = pool.new_value(b3);

    let xw00 = pool.mul(x, w00_handle);
    let h0_sum = pool.add(xw00, b0_handle);
    let h0 = tanh(&mut pool, h0_sum);
    let xw10 = pool.mul(x, w10_handle);
    let h1_sum = pool.add(xw10, b1_handle);
    let h1 = tanh(&mut pool, h1_sum);
    let h0w20 = pool.mul(h0, w20_handle);
    let h1w21 = pool.mul(h1, w21_handle);
    let n20_sum = pool.add(h0w20, h1w21);
    let n20_sum = pool.add(n20_sum, b2_handle);
    let n20 = tanh(&mut pool, n20_sum);
    let h0w30 = pool.mul(h0, w30_handle);
    let h1w31 = pool.mul(h1, w31_handle);
    let n21_sum = pool.add(h0w30, h1w31);
    let n21_sum = pool.add(n21_sum, b3_handle);
    let n21 = tanh(&mut pool, n21_sum);
    let out = pool.add(n20, n21);

    pool.backpropogate(out);

    let num_x = numerical_grad(|v| fwd_2layer2_neurons(v, w00, b0, w10, b1, w20, w21, b2, w30, w31, b3), xv);
    let num_w00 = numerical_grad(|v| fwd_2layer2_neurons(xv, v, b0, w10, b1, w20, w21, b2, w30, w31, b3), w00);
    let num_w10 = numerical_grad(|v| fwd_2layer2_neurons(xv, w00, b0, v, b1, w20, w21, b2, w30, w31, b3), w10);

    assert!((pool.get_grad(x) - num_x).abs() < 1e-4);
    assert!((pool.get_grad(w00_handle) - num_w00).abs() < 1e-4);
    assert!((pool.get_grad(w10_handle) - num_w10).abs() < 1e-4);
}

#[test]
fn test_sub_backward() {
    let mut pool = Pool::new();
    let a = pool.new_value(5.0);
    let b = pool.new_value(3.0);
    let out = pool.sub(a, b);

    pool.backpropogate(out);

    assert_eq!(pool.get_grad(a), 1.0);
    assert_eq!(pool.get_grad(b), -1.0);
}

#[test]
fn test_pow2_backward() {
    let mut pool = Pool::new();
    let a = pool.new_value(2.0);
    let out = pool.pow2(a);

    pool.backpropogate(out);

    assert!((pool.get_grad(a) - 4.0).abs() < 1e-9);
}

#[test]
fn test_mlp_eval_runs_and_flushes_temporary_nodes() {
    let hyperparameters = Hyperparameters {
        learning_rate: 0.05,
        loss_threshold: 0.0,
        batch_size: 32,
        epochs: 10,
    };
    let mut mlp = MLP::new(3, vec![4, 4, 1], Tanh, Tanh, hyperparameters);
    let parameter_count = mlp.pool.param_end;

    let output = mlp.eval(&[2.0, 3.0, 4.0]);

    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());
    assert_eq!(mlp.pool.nodes.len(), parameter_count);
}

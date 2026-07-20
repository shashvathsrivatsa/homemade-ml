use micrograd::*;

fn backprop(v: &mut Value) {
    v.set_grad(1.0);
    v.backpropogate();
}

fn numerical_grad(f_of_x: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-5;
    (f_of_x(x + h) - f_of_x(x - h)) / (2.0 * h)
}

// ——— Basic ops ——————————————————————————————————————————————————————————————

#[test]
fn test_add_backward() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let mut c = &a + &b;
    backprop(&mut c);
    assert_eq!(a.get_grad(), 1.0);
    assert_eq!(b.get_grad(), 1.0);
}

#[test]
fn test_mul_backward() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let mut c = &a * &b;
    backprop(&mut c);
    // dc/da = b = 3,  dc/db = a = 2
    assert_eq!(a.get_grad(), 3.0);
    assert_eq!(b.get_grad(), 2.0);
}

#[test]
fn test_tanh_backward() {
    let a = Value::new(0.5);
    let mut t = a.tanh();
    backprop(&mut t);
    let tv = t.data;
    let expected = 1.0 - tv * tv;
    assert!((a.get_grad() - expected).abs() < 1e-9, "tanh grad: got {}", a.get_grad());
}

#[test]
fn test_chain_mul_tanh() {
    // f = tanh(a * b),  df/da = (1 - tanh(a*b)^2) * b
    let a = Value::new(0.5);
    let b = Value::new(2.0);
    let prod = &a * &b;
    let mut out = prod.tanh();
    backprop(&mut out);

    let dtanh = 1.0 - out.data * out.data;
    assert!((a.get_grad() - dtanh * 2.0).abs() < 1e-9);
    assert!((b.get_grad() - dtanh * 0.5).abs() < 1e-9);
}

// ——— Diamond: shared LEAF ——————————————————————————————————————————————————
//
// f = a*b + a*c  =>  df/da = b+c = 7
//
// `a` is a leaf cloned into two separate parent lists.
// grad is Rc<RefCell<f64>>, so both clones share the same cell.
// set_grad() (i.e. +=) runs BEFORE the visited-check short-circuits
// recursion, so accumulation should still land for leaf nodes.

#[test]
fn test_leaf_diamond() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let c = Value::new(4.0);
    let ab = &a * &b;
    let ac = &a * &c;
    let mut f = ab + ac;
    backprop(&mut f);

    assert_eq!(a.get_grad(), 7.0, "df/da = b+c = 7, got {}", a.get_grad());
    assert_eq!(b.get_grad(), 2.0);
    assert_eq!(c.get_grad(), 2.0);
}

// ——— Diamond: shared NON-LEAF ——————————————————————————————————————————————
//
// x = a + b   (non-leaf computed value)
// f = x*c + x*d  =>  df/da = df/db = c+d = 9
//
// Predicted bug: backprop visits x through the x*c path first, marks x.id
// visited, propagates into a and b with only c's contribution.  The x*d path
// later calls += on x.grad (correct, Rc shared) but then hits `return` before
// propagating the extra d contribution into a and b.
// → a.grad and b.grad end up as 4 (only c), not 9 (c+d).

#[test]
fn test_nonleaf_diamond() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let c = Value::new(4.0);
    let d = Value::new(5.0);

    let x  = &a + &b;     // x = 5
    let xc = &x * &c;     // 20
    let xd = &x * &d;     // 25
    let mut f = xc + xd;  // 45

    backprop(&mut f);

    // c and d are fine — no sharing
    assert_eq!(c.get_grad(), 5.0, "df/dc = x = 5, got {}", c.get_grad());
    assert_eq!(d.get_grad(), 5.0, "df/dd = x = 5, got {}", d.get_grad());

    // These reveal the bug: expected c+d=9, backprop will give 4 (only c's path)
    assert_eq!(a.get_grad(), 9.0, "df/da = c+d = 9, got {}", a.get_grad());
    assert_eq!(b.get_grad(), 9.0, "df/db = c+d = 9, got {}", b.get_grad());
}

// ——— Baseline MLP: 1 input → [2] → 1 output (no non-leaf sharing) ————————
//
// h0 = tanh(x * w00 + b0)
// h1 = tanh(x * w10 + b1)
// out = tanh(h0 * w20 + h1 * w21 + b2)
//
// x is a shared LEAF (fine). h0, h1 are each used by exactly ONE layer-2
// neuron (no sharing), so this topology is clean.  Analytic grads should
// match numeric here.

fn build_1in_2h_1out(
    xv: f64,
    w00: f64, b0: f64,
    w10: f64, b1: f64,
    w20: f64, w21: f64, b2: f64,
) -> (Value, Value, Value, Value, Value, Value, Value) {
    let x   = Value::new(xv);
    let w00 = Value::new(w00);
    let b0  = Value::new(b0);
    let w10 = Value::new(w10);
    let b1  = Value::new(b1);
    let w20 = Value::new(w20);
    let w21 = Value::new(w21);
    let b2  = Value::new(b2);

    let h0  = (&x * &w00 + b0).tanh();
    let h1  = (&x * &w10 + b1).tanh();
    let out = (&h0 * &w20 + &h1 * &w21 + b2).tanh();

    (out, x, w00, w10, w20, w21, Value::new(0.0) /* placeholder */)
}

#[test]
fn test_mlp_single_output_neuron_no_sharing() {
    let (xv, w00v, b0v, w10v, b1v, w20v, w21v, b2v) =
        (0.5, 0.3, 0.1, -0.4, 0.2, 0.6, -0.2, 0.05);

    let (mut out, x, w00, w10, w20, w21, _) =
        build_1in_2h_1out(xv, w00v, b0v, w10v, b1v, w20v, w21v, b2v);
    backprop(&mut out);

    let tol = 1e-4;

    let num_x = numerical_grad(
        |xp| build_1in_2h_1out(xp, w00v, b0v, w10v, b1v, w20v, w21v, b2v).0.data,
        xv,
    );
    let num_w00 = numerical_grad(
        |wp| build_1in_2h_1out(xv, wp, b0v, w10v, b1v, w20v, w21v, b2v).0.data,
        w00v,
    );
    let num_w10 = numerical_grad(
        |wp| build_1in_2h_1out(xv, w00v, b0v, wp, b1v, w20v, w21v, b2v).0.data,
        w10v,
    );
    let num_w20 = numerical_grad(
        |wp| build_1in_2h_1out(xv, w00v, b0v, w10v, b1v, wp, w21v, b2v).0.data,
        w20v,
    );
    let num_w21 = numerical_grad(
        |wp| build_1in_2h_1out(xv, w00v, b0v, w10v, b1v, w20v, wp, b2v).0.data,
        w21v,
    );

    assert!((x.get_grad()   - num_x  ).abs() < tol, "df/dx:   analytic={:.6} numeric={:.6}", x.get_grad(),   num_x);
    assert!((w00.get_grad() - num_w00).abs() < tol, "df/dw00: analytic={:.6} numeric={:.6}", w00.get_grad(), num_w00);
    assert!((w10.get_grad() - num_w10).abs() < tol, "df/dw10: analytic={:.6} numeric={:.6}", w10.get_grad(), num_w10);
    assert!((w20.get_grad() - num_w20).abs() < tol, "df/dw20: analytic={:.6} numeric={:.6}", w20.get_grad(), num_w20);
    assert!((w21.get_grad() - num_w21).abs() < tol, "df/dw21: analytic={:.6} numeric={:.6}", w21.get_grad(), num_w21);
}

// ——— MLP with TWO neurons in layer 2 (non-leaf sharing) ——————————————————
//
// h0 = tanh(x * w00 + b0)
// h1 = tanh(x * w10 + b1)
//   -- layer 2 has TWO neurons, so h0 and h1 become shared non-leaf nodes --
// n20 = tanh(h0 * w20 + h1 * w21 + b2)
// n21 = tanh(h0 * w30 + h1 * w31 + b3)
// out = n20 + n21
//
// The visited-check will mark h0 and h1 after processing n20's path.
// When n21's path reaches its clones of h0/h1, += still accumulates grad,
// but the recursive descent into h0/h1's parents (w00, w10, x) is skipped.
// → w00, w10 and x get INCOMPLETE gradients from only one of the two paths.

fn fwd_2layer2_neurons(
    xv: f64,
    w00: f64, b0: f64,
    w10: f64, b1: f64,
    w20: f64, w21: f64, b2: f64,
    w30: f64, w31: f64, b3: f64,
) -> f64 {
    let x  = Value::new(xv);
    let h0 = (&x * &Value::new(w00) + Value::new(b0)).tanh();
    let h1 = (&x * &Value::new(w10) + Value::new(b1)).tanh();
    let n20 = (&h0 * &Value::new(w20) + &h1 * &Value::new(w21) + Value::new(b2)).tanh();
    let n21 = (&h0 * &Value::new(w30) + &h1 * &Value::new(w31) + Value::new(b3)).tanh();
    (n20 + n21).data
}

#[test]
fn test_mlp_two_layer2_neurons_numeric_grad() {
    let (xv, w00v, b0v, w10v, b1v) = (0.5, 0.3, 0.1, -0.4, 0.2);
    let (w20v, w21v, b2v) = (0.6, -0.2, 0.05);
    let (w30v, w31v, b3v) = (-0.5, 0.7, -0.1);

    // Build graph and run backprop
    let x   = Value::new(xv);
    let w00 = Value::new(w00v);
    let w10 = Value::new(w10v);
    let h0  = (&x * &w00 + Value::new(b0v)).tanh();
    let h1  = (&x * &w10 + Value::new(b1v)).tanh();
    let n20 = (&h0 * &Value::new(w20v) + &h1 * &Value::new(w21v) + Value::new(b2v)).tanh();
    let n21 = (&h0 * &Value::new(w30v) + &h1 * &Value::new(w31v) + Value::new(b3v)).tanh();
    let mut out = n20 + n21;
    backprop(&mut out);

    let tol = 1e-4;

    let num_x = numerical_grad(
        |xp| fwd_2layer2_neurons(xp, w00v, b0v, w10v, b1v, w20v, w21v, b2v, w30v, w31v, b3v),
        xv,
    );
    let num_w00 = numerical_grad(
        |wp| fwd_2layer2_neurons(xv, wp, b0v, w10v, b1v, w20v, w21v, b2v, w30v, w31v, b3v),
        w00v,
    );
    let num_w10 = numerical_grad(
        |wp| fwd_2layer2_neurons(xv, w00v, b0v, wp, b1v, w20v, w21v, b2v, w30v, w31v, b3v),
        w10v,
    );

    // Print what we got regardless — the assert messages will tell the story
    println!("df/dx:   analytic={:.6}  numeric={:.6}  diff={:.6}", x.get_grad(),   num_x,   (x.get_grad()   - num_x).abs());
    println!("df/dw00: analytic={:.6}  numeric={:.6}  diff={:.6}", w00.get_grad(), num_w00, (w00.get_grad() - num_w00).abs());
    println!("df/dw10: analytic={:.6}  numeric={:.6}  diff={:.6}", w10.get_grad(), num_w10, (w10.get_grad() - num_w10).abs());

    assert!((x.get_grad()   - num_x  ).abs() < tol, "df/dx:   analytic={:.6} numeric={:.6}", x.get_grad(),   num_x);
    assert!((w00.get_grad() - num_w00).abs() < tol, "df/dw00: analytic={:.6} numeric={:.6}", w00.get_grad(), num_w00);
    assert!((w10.get_grad() - num_w10).abs() < tol, "df/dw10: analytic={:.6} numeric={:.6}", w10.get_grad(), num_w10);
}

// ——— Sub backward ——————————————————————————————————————————————————————————
// f = a - b  =>  df/da = 1,  df/db = -1
// "-" currently falls through to `_ => 0.0` so this will catch missing impl.

#[test]
fn test_sub_backward() {
    let a = Value::new(5.0);
    let b = Value::new(3.0);
    let mut f = &a - &b;
    backprop(&mut f);
    assert_eq!(a.get_grad(),  1.0, "df/da should be 1, got {}", a.get_grad());
    assert_eq!(b.get_grad(), -1.0, "df/db should be -1, got {}", b.get_grad());
}

// ——— Pow2 backward —————————————————————————————————————————————————————————
// f = a^2  =>  df/da = 2*a = 4  (at a=2)

#[test]
fn test_pow2_backward() {
    let mut a = Value::new(2.0);
    let mut f = a.pow2();
    backprop(&mut f);
    assert!((a.get_grad() - 4.0).abs() < 1e-9, "df/da should be 4, got {}", a.get_grad());
}

// ——— End-to-end MLP struct ————————————————————————————————————————————————
// Runs the actual MLP::new / MLP::call / backpropogate path to make sure the
// struct wiring produces non-zero, finite gradients for all inputs.
// Uses a fixed seed by constructing weights manually isn't possible, so we
// just assert the gradients are finite and at least one is non-zero.

#[test]
fn test_mlp_struct_backprop_runs() {
    let x: Vec<Value> = (1..4).map(|i| Value::new((i + 1) as f64)).collect();
    let m = MLP::new(3, vec![4, 4, 1]);
    let mut out = m.call(&x)[0].clone();
    out.backpropogate();

    // output grad is set inside backpropogate
    assert!(out.data.is_finite());
    assert!(out.get_grad() == 1.0);

    // every input should have a finite, non-zero gradient
    for (i, xi) in x.iter().enumerate() {
        assert!(xi.get_grad().is_finite(), "x[{}] grad is not finite", i);
        assert!(xi.get_grad() != 0.0,      "x[{}] grad is zero",       i);
    }
}

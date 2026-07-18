use micrograd::*;


// ——— Main ———————————————————————————————————————————————————————————————————————————————————————————————————————————

fn main() {

    // inputs
    let x1 = Value::new("x1", 2.0);
    let x2 = Value::new("x2", 0.0);

    // weights
    let w1 = Value::new("w1", -3.0);
    let w2 = Value::new("w2", 1.0);

    // bias
    let b = Value::new("b", 6.8813735870195432);

    // output
    let x1w1 = (&x1 * &w1).label("x1w1");
    let x2w2 = (&x2 * &w2).label("x2w2");
    let x1w1x2w2 = (&x1w1 + &x2w2).label("x1w1x2w2");

    let n = (&x1w1x2w2 + &b).label("n");
    let mut o = n.tanh().label("o");

    // backpropogate
    o.set_grad(1.0);
    let mut visited: HashSet<usize> = HashSet::new();
    o.backpropogate(&mut visited);

    // draw
    draw_dot(&o);
}


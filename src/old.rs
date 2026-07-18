
    // let xs: Vec<f64> = range(-5.0, 5.0, 0.25);
    // let ys: Vec<f64> = xs.iter().map(|&x| f(x)).collect();
    // plot(&xs, &ys);

    // let h = 0.000001;

    // let a = 2.0;
    // let b = -3.0;
    // let c = 10.0;

    // let d1 = a * b + c;
    // let d2 = a * b + (c + h);

    // println!("{}", (d2 - d1) / h);

fn f(x: f64) -> f64 {
    3.0 * x.powi(2) - 4.0 * x + 5.0
}

    // let mut a = Value::new("a", 2.0);
    // let mut b = Value::new("b", -3.0);
    // let mut c = Value::new("c", 10.0);
    // let mut d = (&a * &b).label("d");
    // let mut e = (&d + &c).label("e");
    // let mut f = Value::new("f", -2.0);
    // let mut l = (&e * &f).label("L");

    // l.set_grad(1.0);
    // f.set_grad(e.data * l.get_grad());
    // e.set_grad(f.data * l.get_grad());
    // d.set_grad(1.0 * e.get_grad());
    // c.set_grad(1.0 * e.get_grad());
    // b.set_grad(a.data * d.get_grad());
    // a.set_grad(b.data * d.get_grad());

    // draw_dot(&l);

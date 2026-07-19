use crate::*;


// ——— Neuron —————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(n_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            w: (0..n_inputs)
                .map(|_| Value::new(rng.gen_range(-1.0..1.0)))
                .collect(),
            b: Value::new(rng.gen_range(-1.0..1.0)),
        }
    }

    pub fn call(&self, x: &[Value]) -> Value {
        let sum = x.iter().zip(self.w.iter())
            .map(|(xi, wi)| xi * wi)
            .fold(self.b.clone(), |acc, xiwi| acc + xiwi);
        sum.tanh()
    }
}


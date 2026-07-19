use crate::*;


// ——— Layer ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Layer {
    pub neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(n_inputs: usize, n_outputs: usize) -> Self {
        Self {
            neurons: (0..n_outputs).map(|_| Neuron::new(n_inputs)).collect()
        }
    }

    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }
}


use crate::*;


// ——— MLP ————————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(n_inputs: usize, n_outputs: Vec<usize>) -> Self {
        let prev = |y: usize| if y == 0 { n_inputs } else { n_outputs[y - 1] };
        Self {
            layers: (0..n_outputs.len())
                .map(|n_outputs_i| Layer::new(prev(n_outputs_i), n_outputs[n_outputs_i]))
                .collect(),
        }
    }

    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        self.layers.iter().fold(x.to_vec(), |acc, layer| layer.call(&acc))
    }
}


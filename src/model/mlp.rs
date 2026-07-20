use crate::*;


// ——— Neuron —————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    fn new(n_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            w: (0..n_inputs)
                .map(|_| Value::new(rng.gen_range(-1.0..1.0)))
                .collect(),
                b: Value::new(rng.gen_range(-1.0..1.0)),
        }
    }

    fn call(&self, x: &[Value]) -> Value {
        let sum = x.iter().zip(self.w.iter())
            .map(|(xi, wi)| xi * wi)
            .fold(self.b.clone(), |acc, xiwi| acc + xiwi);
        sum.tanh()
    }

    fn parameters(&mut self) -> Vec<&mut Value> {
        self.w.iter_mut().chain(std::iter::once(&mut self.b)).collect()
    }
}


// ——— Layer ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Layer {
    pub neurons: Vec<Neuron>
}

impl Layer {
    fn new(n_inputs: usize, n_outputs: usize) -> Self {
        Self {
            neurons: (0..n_outputs).map(|_| Neuron::new(n_inputs)).collect()
        }
    }

    fn call(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }

    fn parameters(&mut self) -> Vec<&mut Value> {
        self.neurons.iter_mut().flat_map(|neuron| neuron.parameters()).collect()
    }
}


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

    pub fn parameters(&mut self) -> Vec<&mut Value> {
        self.layers.iter_mut().flat_map(|layer| layer.parameters()).collect()
    }

    pub fn train(&mut self, xs: &[Vec<Value>], ys: &[Value], hyperparameters: &Hyperparameters) {

        let mut y_pred: Vec<Value> = xs.iter().map(|x| self.call(&x)[0].clone()).collect();

        let mut loss = ys.iter().zip(y_pred.iter()).fold(
            Value::new(0.0), |acc, (ygt, yout)| acc + (ygt - yout).pow2()
        );
        println!("Loss: {:?}", loss);

        // Gradient descent
        let mut iterations = 1;
        while loss.data > 0.01 {

            // Backprop
            loss.backpropogate();

            // Update
            self.parameters().iter_mut().for_each(|p| p.data -= hyperparameters.learning_rate * p.get_grad());
            y_pred = xs.iter().map(|x| self.call(&x)[0].clone()).collect();

            loss = ys.iter().zip(y_pred.iter()).fold(
                Value::new(0.0), |acc, (ygt, yout)| acc + (ygt - yout).pow2()
            );
            println!("Loss: {:?}", loss);

            iterations += 1;
        }

        // Visualize
        loss.backpropogate();
        println!("{:?}", y_pred);
        println!("{}", iterations);
        draw_dot(&loss);
    }

    pub fn eval(&self, x: &[Value]) -> Vec<Value> {
        self.call(&x)
    }
}


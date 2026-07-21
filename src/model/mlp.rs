use crate::*;


// ——— Neuron —————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    fn new(pool: &mut Pool, n_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            w: (0..n_inputs)
                .map(|_| pool.new_value(rng.gen_range(-1.0..1.0)))
                .collect(),
                b: pool.new_value(rng.gen_range(-1.0..1.0)),
        }
    }

    fn call(&self, pool: &mut Pool, x: &[Value]) -> Value {
        let products: Vec<Value> = x.iter().zip(self.w.iter())
            .map(|(&xi, &wi)| pool.mul(xi, wi))
            .collect();

        let sum = products.into_iter().fold(self.b, |acc, xiwi| pool.add(acc, xiwi));
        pool.tanh(sum)
    }

    fn parameters(&self) -> Vec<Value> {
        self.w.iter().copied().chain(std::iter::once(self.b)).collect()
    }
}


// ——— Layer ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Layer {
    pub neurons: Vec<Neuron>
}

impl Layer {
    fn new(pool: &mut Pool, n_inputs: usize, n_outputs: usize) -> Self {
        Self {
            neurons: (0..n_outputs).map(|_| Neuron::new(pool, n_inputs)).collect()
        }
    }

    fn call(&self, pool: &mut Pool, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(pool, x)).collect()
    }

    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|neuron| neuron.parameters()).collect()
    }
}


// ——— MLP ————————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct MLP {
    layers: Vec<Layer>,
    pub pool: Pool,
    hyperparameters: Hyperparameters,
}

impl MLP {
    pub fn new(n_inputs: usize, n_outputs: Vec<usize>) -> Self {
        let mut pool = Pool::new();
        let prev = |y: usize| if y == 0 { n_inputs } else { n_outputs[y - 1] };

        let layers = (0..n_outputs.len())
                .map(|n_outputs_i| Layer::new(&mut pool, prev(n_outputs_i), n_outputs[n_outputs_i]))
                .collect();

        pool.set_param_end();

        Self { layers, pool, hyperparameters: Hyperparameters::new() }
    }

    fn call(&mut self, x: &[Value]) -> Vec<Value> {
        self.layers.iter().fold(x.to_vec(), |acc, layer| layer.call(&mut self.pool, &acc))
    }

    pub fn parameters(&mut self) -> Vec<Value> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }

    pub fn train(&mut self, xs: &[Vec<Value>], ys: &[Value]) {
        let s = Instant::now();
        let mut steps = 1;

        loop {
            steps += 1;
            self.pool.flush();

            // Calculate
            let y_pred: Vec<Value> = xs.iter().map(|x| self.call(&x)[0]).collect();
            let loss = mse_loss(&mut self.pool, ys, &y_pred);
            print!("Loss: "); self.pool.print(loss);

            // Break if converges
            if self.pool.get_data(loss) < 0.01 {
                println!("{} steps", steps);
                println!("{:.2?}", s.elapsed());
                draw_dot(&self.pool, loss);
                break;
            }

            // Backprop
            self.pool.backpropogate(loss);

            // Update
            self.parameters().iter().for_each(|&p| {
                self.pool.update(p, self.hyperparameters.learning_rate)
            });
        }
    }

    pub fn eval(&mut self, x: &[Value]) -> Vec<Value> {
        self.call(&x)
    }
}


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
        sum
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

    fn call(&self, pool: &mut Pool, x: &[Value], activation: &Activation) -> Vec<Value> {
        let logits: Vec<Value> = self.neurons.iter().map(|n| n.call(pool, x)).collect();
        activation.apply(pool, &logits)
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
    hidden_activation: Activation,
    output_activation: Activation,
}

impl MLP {
    pub fn new(
        n_inputs: usize,
        n_outputs: Vec<usize>,
        hidden_activation: Activation,
        output_activation: Activation,
        hyperparameters: Hyperparameters,
    ) -> Self {
        let mut pool = Pool::new();
        let prev = |y: usize| if y == 0 { n_inputs } else { n_outputs[y - 1] };

        let layers = (0..n_outputs.len())
            .map(|n_outputs_i| Layer::new(&mut pool, prev(n_outputs_i), n_outputs[n_outputs_i]))
            .collect();

        pool.set_param_end();

        Self { layers, pool, hyperparameters, hidden_activation, output_activation }
    }

    fn call(&mut self, x: &[Value]) -> Vec<Value> {
        self.layers.iter().enumerate().fold(x.to_vec(), |acc, (i, layer)| {
            let activation = if i == self.layers.len() - 1 { &self.output_activation } else { &self.hidden_activation };
            layer.call(&mut self.pool, &acc, activation)
        })
    }

    pub fn train_batch(&mut self, xs: &[Vec<f64>], ys: &[f64]) {
        let s = Instant::now();

        for epoch in 0..self.hyperparameters.epochs {
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..xs.len()).collect();
            indices.shuffle(&mut rng);

            for (batch_num, chunk) in indices.chunks(self.hyperparameters.batch_size).enumerate() {
                let xs: Vec<Vec<Value>> = chunk.iter().map(|&i| {
                    xs[i].iter().map(|&input| self.pool.new_value(input)).collect()
                }).collect();
                let ys: Vec<Value> = chunk.iter().map(|&i| self.pool.new_value(ys[i])).collect();

                // Loss
                let y_pred: Vec<Vec<Value>> = xs.iter().map(|x| self.call(&x)).collect();
                let loss = cross_entropy_loss(&mut self.pool, &ys, &y_pred);
                print!(
                    "[{}/{}] [{}/{}] Loss: ",
                    batch_num,
                    indices.len().div_ceil(self.hyperparameters.batch_size),
                    epoch + 1,
                    self.hyperparameters.epochs,
                ); self.pool.print(loss);

                // Backprop
                self.pool.backpropogate(loss);

                // Update
                self.parameters().iter().for_each(|&p| {
                    self.pool.update(p, self.hyperparameters.learning_rate)
                });

                self.pool.flush();
            }
        }

        println!("{:.2?}", s.elapsed());
    }

    pub fn train_full(&mut self, xs: &[Vec<f64>], ys: &[f64]) {
        let s = Instant::now();
        let mut steps = 0;

        let xs: Vec<Vec<Value>> = xs.iter().map(|row| {
            row.iter().map(|&entry| self.pool.new_value(entry)).collect()
        }).collect();
        let ys: Vec<Value> = ys.iter().map(|&entry| self.pool.new_value(entry)).collect();
        self.pool.set_input_end();

        loop {
            steps += 1;
            self.pool.flush_compute();

            // Calculate
            let y_pred: Vec<Vec<Value>> = xs.iter().map(|x| self.call(&x)).collect();
            let loss = cross_entropy_loss(&mut self.pool, &ys, &y_pred);
            print!("Loss: "); self.pool.print(loss);

            // Break if converges
            if self.pool.get_data(loss) < 0.01 {
                println!("{} steps", steps);
                break;
            }

            // Backprop
            self.pool.backpropogate(loss);

            // Update
            self.parameters().iter().for_each(|&p| {
                self.pool.update(p, self.hyperparameters.learning_rate)
            });
        }

        println!("{:.2?}", s.elapsed());
        self.pool.flush();
    }

    pub fn eval(&mut self, x: &[f64]) -> Vec<f64> {
        let x: Vec<Value> = x.iter().map(|&entry| self.pool.new_value(entry)).collect();
        let result: Vec<f64> = self.call(&x).iter().map(|&v| self.pool.get_data(v)).collect();
        self.pool.flush();
        result
    }

    pub fn parameters(&mut self) -> Vec<Value> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }

    pub fn save(&mut self) {
        let weights: Vec<f64> = self.parameters().iter().map(|&p| self.pool.get_data(p)).collect();
        let txt = weights.iter().map(|w| w.to_string()).collect::<Vec<_>>().join("\n");
        fs::write("model.txt", txt).unwrap();
    }

    pub fn load(&mut self) {
        let txt = fs::read_to_string("model.txt").unwrap();
        let weights: Vec<f64> = txt.lines().map(|l| l.parse().unwrap()).collect();
        self.parameters().iter().zip(weights.iter()).for_each(|(&p, &w)| {
            self.pool.set_data(p, w);
        });
    }
}


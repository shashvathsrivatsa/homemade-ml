use crate::*;


// ——— Layer ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Layer {
    pub w: Tensor,
    pub b: Tensor,
}

impl Layer {
    fn new(pool: &mut TensorPool, n_inputs: usize, n_outputs: usize) -> Self {
        let w = pool.new_rand(vec![n_inputs, n_outputs]);
        let b = pool.new_zeros(vec![n_outputs]);
        Self { w, b }
    }

    fn call(&self, pool: &mut TensorPool, x: Tensor, activation: &Activation) -> Tensor {
        let c = pool.matmul(x, self.w);
        let z = pool.bias_add(c, self.b);
        z
        // TODO: activation
    }

    fn parameters(&self, pool: &mut TensorPool) -> Vec<Tensor> {
        vec![self.w, self.b]
    }
}


// ——— MLP ————————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct MLP {
    layers: Vec<Layer>,
    pub pool: TensorPool,
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
        let mut pool = TensorPool::new();
        let prev = |y: usize| if y == 0 { n_inputs } else { n_outputs[y - 1] };

        let layers = (0..n_outputs.len())
            .map(|n_outputs_i| Layer::new(&mut pool, prev(n_outputs_i), n_outputs[n_outputs_i]))
            .collect();

        Self { layers, pool, hyperparameters, hidden_activation, output_activation }
    }

    // —— Propogate ————————————————————————————————————————————————————————————————————————
    fn call(&mut self, x: Tensor) -> Tensor {
        self.layers.iter().enumerate().fold(x, |acc, (i, layer)| {
            let activation = if i == self.layers.len() - 1 { &self.output_activation } else { &self.hidden_activation };
            layer.call(&mut self.pool, acc, activation)
        })
    }

    pub fn parameters(&mut self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|layer| layer.parameters(&mut self.pool)).collect()
    }

    // —— Training —————————————————————————————————————————————————————————————————————————
    pub fn train_batch(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let s = Instant::now();

        for epoch in 0..self.hyperparameters.epochs {
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..x.len()).collect();
            indices.shuffle(&mut rng);

            for (batch_num, chunk) in indices.chunks(self.hyperparameters.batch_size).enumerate() {
                // TODO: (flush)

                // Load
                let x_data: Vec<f64> = chunk.iter().flat_map(|&i| x[i]).collect();
                let y_data: Vec<f64> = chunk.iter().map(|&i| y[i]).collect();
                let x = self.pool.new_tensor(x_data, vec![chunk.len(), x[0].len()]);
                let y = self.pool.new_tensor(y_data, vec![chunk.len()]);

                // Forward pass
                let y_pred: Tensor = self.call(x);

                // Compute loss
                let loss = cross_entropy_loss(&mut self.pool, y, y_pred);
                print!(
                    "[{}/{}] [{}/{}] Loss: ",
                    batch_num,
                    indices.len().div_ceil(self.hyperparameters.batch_size),
                    epoch + 1,
                    self.hyperparameters.epochs,
                ); self.pool.print(loss);

                // Backprop
                self.pool.backpropogate(loss);

                // Update weights
                self.parameters().iter().for_each(|&p| {
                    self.pool.update(p, self.hyperparameters.learning_rate)
                });
            }
        }

        println!("{:.2?}", s.elapsed());
    }

    // —— Testing ——————————————————————————————————————————————————————————————————————————
    pub fn eval(&mut self, x: &[f64]) -> Vec<f64> {
        let x: Vec<Value> = x.iter().map(|&entry| self.pool.new_value(entry)).collect();
        let result: Vec<f64> = self.call(&x).iter().map(|&v| self.pool.get_data(v)).collect();
        self.pool.flush();
        result
    }

    pub fn test(&mut self, xs: &[Vec<f64>], ys: &[f64]) -> f64 {
        xs.iter().enumerate().fold(0.0, |total_correct, (i, x)| {
            let y = self.eval(x);
            let y_pred = (0..=9).fold(0, |max_i, i| if y[i] > y[max_i] { i } else { max_i });
            if i % 100 == 0 { println!("{:.2}% ({}/{})", i as f64 / xs.len() as f64 * 100.0, i, xs.len()) }
            if y_pred as f64 == ys[i] { total_correct + 1.0 } else { total_correct }
        }) / xs.len() as f64
    }

    // —— Store ————————————————————————————————————————————————————————————————————————————
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


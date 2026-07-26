use crate::*;


// ——— Layer ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Layer {
    pub w: Tensor,
    pub b: Tensor,
}

impl Layer {
    fn new(pool: &mut Pool, n_inputs: usize, n_outputs: usize) -> Self {
        let w = pool.new_rand(vec![n_inputs, n_outputs]);
        let b = pool.fill(vec![n_outputs], 0.0);
        Self { w, b }
    }

    fn call(&self, pool: &mut Pool, x: Tensor, activation: &Activation) -> Tensor {
        let c = pool.matmul(x, self.w);
        let z = pool.bias_add(c, self.b);
        activation.apply(pool, z)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.w, self.b]
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

    // —— Propogate ————————————————————————————————————————————————————————————————————————
    fn call(&mut self, x: Tensor) -> Tensor {
        self.layers.iter().enumerate().fold(x, |acc, (i, layer)| {
            let activation = if i == self.layers.len() - 1 { &self.output_activation } else { &self.hidden_activation };
            layer.call(&mut self.pool, acc, activation)
        })
    }

    pub fn parameters(&mut self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }

    // —— Training —————————————————————————————————————————————————————————————————————————
    pub fn train_batch(&mut self, x: &[Vec<f32>], y: &[f32]) {
        let s = Instant::now();

        for epoch in 0..self.hyperparameters.epochs {
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..x.len()).collect();
            indices.shuffle(&mut rng);

            for (batch_num, chunk) in indices.chunks(self.hyperparameters.batch_size).enumerate() {

                // TODO:
                let w = self.parameters()[0];
                let w1 = self.pool.get_grad(w)[0];
                println!("{}", w1);

                // Load
                let x_data: Vec<f32> = chunk.iter().flat_map(|&i| x[i].iter().copied()).collect();
                let y_data: Vec<f32> = chunk.iter().map(|&i| y[i]).collect();
                let x = self.pool.new_tensor(x_data, vec![chunk.len(), x[0].len()]);
                let y = self.pool.new_tensor(y_data, vec![chunk.len()]);

                // Forward pass
                let y_pred = self.call(x);

                // Compute loss
                let loss = cross_entropy_loss(&mut self.pool, y_pred, y);
                self.pool.print(loss);

                // Log
                let total_batches = indices.len().div_ceil(self.hyperparameters.batch_size);
                let percent = (epoch * total_batches + batch_num + 1) as f32
                    / (total_batches * self.hyperparameters.epochs) as f32;

                // print!(
                //     "[{}>{}] {:.2}% | Epoch: {}/{} | Loss: {:.2} {:1}\r",
                //     "=".repeat((percent * 30.0) as usize),
                //     " ".repeat(((1.0 - percent) * 30.0) as usize),
                //     percent * 100.0,
                //     epoch + 1,
                //     self.hyperparameters.epochs,
                //     self.pool.get_data(loss)[0],
                //     "",
                // );
                // std::io::stdout().flush().unwrap();

                // Backprop
                self.pool.backpropogate(loss);

                // Update weights
                self.parameters().iter().for_each(|&p| {
                    self.pool.update(p, self.hyperparameters.learning_rate)
                });

                // Flush
                self.pool.flush();
            }
        }

        print!("\r{:70}\r", "");
        println!("{:.2?}", s.elapsed());
    }

    // —— Testing ——————————————————————————————————————————————————————————————————————————
    pub fn eval(&mut self, x: &[f32]) -> Vec<f32> {
        let x = self.pool.new_tensor(x.to_owned(), vec![1, x.len()]);
        let y = self.call(x);
        let result: Vec<f32> = self.pool.get_data(y).to_owned();
        self.pool.flush();
        result
    }

    pub fn test(&mut self, xs: &[Vec<f32>], ys: &[f32]) -> f32 {
        let total_correct = xs.iter().enumerate().fold(0.0, |total_correct, (i, x)| {
            let y = self.eval(x);
            let y_pred = (0..=9).fold(0, |max_i, i| if y[i] > y[max_i] { i } else { max_i });

            // Print
            if i % 10 == 0 {
                let filled = (i + 1) * 30 / xs.len();
                print!(
                    "\r[{}>{}] {:.0}%",
                    "=".repeat(filled),
                    " ".repeat(30 - filled),
                    i as f32 / xs.len() as f32 * 100.0
                );
                std::io::stdout().flush().unwrap();
            }

            if y_pred as f32 == ys[i] { total_correct + 1.0 } else { total_correct }
        });

        print!("\r{:50}\r", "");
        total_correct / xs.len() as f32
    }

    // —— Store ————————————————————————————————————————————————————————————————————————————
    pub fn save(&mut self) {
        let parameters = self.parameters();
        let weights: Vec<f32> = parameters.iter()
            .flat_map(|&p| self.pool.get_data(p).to_owned())
            .collect();
        let txt = weights.iter().map(|w| w.to_string()).collect::<Vec<_>>().join("\n");
        fs::write("model.txt", txt).unwrap();
        println!("Saved model weights");
    }

    pub fn load(&mut self) {
        let txt = fs::read_to_string("model.txt").unwrap();
        let weights: Vec<f32> = txt.lines().map(|l| l.parse().unwrap()).collect();
        let mut offset = 0;
        for &p in self.parameters().iter() {
            let size = self.pool.get_data(p).len();
            self.pool.set_data(p, weights[offset..offset + size].to_vec());
            offset += size;
        }
        println!("Loaded model weights");
    }
}


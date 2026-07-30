use crate::*;

// ——— Layer ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Layer {
    pub w: Tensor,
    pub b: Tensor,
}

impl Layer {
    fn new(pool: &mut Pool, n_inputs: usize, n_layers: usize) -> Self {
        let w = pool.new_rand(vec![n_inputs, n_layers]);
        let b = pool.fill(vec![n_layers], 0.0);
        Self { w, b }
    }

    fn call(&self, pool: &mut Pool, x: Tensor, activation: &Activation, dropout_rate: f32) -> Tensor {
        let c = pool.matmul(x, self.w);
        let z = pool.bias_add(c, self.b);
        let y = activation.apply(pool, z);
        pool.dropout(y, dropout_rate)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.w, self.b]
    }
}

// ——— MLP ————————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct MLP {
    pool: Pool,
    layers: Vec<Layer>,
    hyperparameters: MlpHyperparameters,
    optimizer_data: OptimizerData,
}

impl MLP {
    pub fn new(
        hyperparameters: MlpHyperparameters,
    ) -> Self {
        let mut pool = Pool::new();
        let prev = |y: usize| if y == 0 { hyperparameters.n_inputs } else { hyperparameters.n_layers[y - 1] };

        let layers: Vec<Layer> = (0..hyperparameters.n_layers.len())
            .map(|n_outputs_i| Layer::new(&mut pool, prev(n_outputs_i), hyperparameters.n_layers[n_outputs_i]))
            .collect();

        pool.set_param_end();

        let optimizer_data = match hyperparameters.optimizer {
            SGD => OptimizerData::SGD(SGD::new(hyperparameters.lr)),
            AdamOptimizer => {
                let opts = layers.iter().flat_map(|l| [
                    AdamOptimizer::new(hyperparameters.lr, &pool.gpu, pool.nodes[l.w.0].len),
                    AdamOptimizer::new(hyperparameters.lr, &pool.gpu, pool.nodes[l.b.0].len),
                ]).collect();
                OptimizerData::Adam(opts)
            }
        };

        Self {
            layers,
            pool,
            hyperparameters,
            optimizer_data,
        }
    }

    // —— Propogate ————————————————————————————————————————————————————————————————————————
    fn call(&mut self, x: Tensor) -> Tensor {
        self.layers.iter().enumerate().fold(x, |acc, (i, layer)| {
            let (activation, dropout_rate) = if i == self.layers.len() - 1 {
                (&self.hyperparameters.output_activation, 0.0)
            } else {
                (&self.hyperparameters.hidden_activation, self.hyperparameters.dropout_rate)
            };

            layer.call(&mut self.pool, acc, activation, dropout_rate)
        })
    }

    fn parameters(&mut self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    // —— Training —————————————————————————————————————————————————————————————————————————
    pub fn train(&mut self, x: &[Vec<f32>], y: &[Vec<f32>]) {
        match self.hyperparameters.training_mode {
            TrainingMode::Full { loss_threshold } => self.train_full(x, y, loss_threshold),
            TrainingMode::Batch { batch_size, epochs } => self.train_batch(x, y, batch_size, epochs),
            TrainingMode::OnePass => self.train_once(x, y),
        }
    }

    fn train_once(&mut self, x: &[Vec<f32>], y: &[Vec<f32>]) {

        // Load
        let x_data: Vec<f32> = x.iter().flatten().copied().collect();
        let y_data: Vec<f32> = y.iter().flatten().copied().collect();
        let x_input = self.pool.new_tensor(x_data, vec![x.len(), x[0].len()]);
        let y_target = self.pool.new_tensor(y_data, vec![y.len(), y[0].len()]);

        // Do the whole thing
        let y_pred = self.call(x_input);
        let loss = self.hyperparameters.loss_function.apply(&mut self.pool, y_pred, y_target);
        self.pool.backpropogate(loss);
        self.pool.update_weights(&mut self.optimizer_data);
        self.pool.flush();
    }

    fn train_full(&mut self, x: &[Vec<f32>], y: &[Vec<f32>], loss_threshold: f32) {
        println!("Training...");
        let s = Instant::now();
        let y = &y[0];
        let mut steps = 0;
        let mut loss_graph = LossGraph::new().expect("failed to initialize loss graph");

        // Load
        let x_data: Vec<f32> = x.iter().flatten().copied().collect();
        let x_input = self.pool.new_tensor(x_data, vec![x.len(), x[0].len()]);
        let y_target = self.pool.new_tensor(y.to_vec(), vec![y.len()]);

        loop {
            steps += 1;

            // Forward pass
            let y_pred = self.call(x_input);

            // Compute loss
            let loss = self.hyperparameters.loss_function.apply(&mut self.pool, y_pred, y_target);
            let l = self.pool.get_data(loss)[0];

            // Log
            loss_graph
                .draw(steps, l)
                .expect("failed to draw loss graph");

            // End training if loss converges or the user quits
            if l < loss_threshold
                || loss_graph.should_quit().expect("failed to read terminal input")
            {
                self.pool.flush();
                let save_result = loss_graph.save_png("loss.png");
                drop(loss_graph);
                save_result.expect("failed to save loss graph");
                println!("Saved loss graph to loss.png");
                println!("{:.2?}", s.elapsed());
                break;
            }

            // Backprop
            self.pool.backpropogate(loss);

            // Update weights
            self.pool.update_weights(&mut self.optimizer_data);

            // Flush
            self.pool.flush();
        }
    }

    fn train_batch(&mut self, x: &[Vec<f32>], y: &[Vec<f32>], batch_size: usize, epochs: usize) {
        println!("Training...");
        let s = Instant::now();
        let y = &y[0];

        for epoch in 0..epochs {
            let mut rng = thread_rng();
            let mut indices: Vec<usize> = (0..x.len()).collect();
            indices.shuffle(&mut rng);

            for (batch_num, chunk) in indices.chunks(batch_size).enumerate() {

                // Load
                let x_data: Vec<f32> = chunk.iter().flat_map(|&i| x[i].iter().copied()).collect();
                let y_data: Vec<f32> = chunk.iter().map(|&i| y[i]).collect();
                let x = self.pool.new_tensor(x_data, vec![chunk.len(), x[0].len()]);
                let y = self.pool.new_tensor(y_data, vec![chunk.len()]);

                // Forward pass
                let y_pred = self.call(x);

                // Compute loss
                let loss = self.hyperparameters.loss_function.apply(&mut self.pool, y_pred, y);

                // Log
                let total_batches = indices.len().div_ceil(batch_size);
                let percent = (epoch * total_batches + batch_num + 1) as f32
                    / (total_batches * epochs) as f32;

                print!(
                    "[{}>{}] {:.2}% | Epoch: {}/{} {:1}\r",
                    "=".repeat((percent * 30.0) as usize),
                    " ".repeat(((1.0 - percent) * 30.0) as usize),
                    percent * 100.0,
                    epoch + 1,
                    epochs,
                    "",
                );
                std::io::stdout().flush().unwrap();

                // Backprop
                self.pool.backpropogate(loss);

                // Update weights
                self.pool.update_weights(&mut self.optimizer_data);

                // Flush
                self.pool.flush();
            }
        }

        print!("\r{:70}\r", "");
        println!("{:.2?}", s.elapsed());
    }

    // —— Testing ——————————————————————————————————————————————————————————————————————————
    pub fn eval(&mut self, x: &[f32]) -> Vec<f32> {
        let old_dropout_rate = self.hyperparameters.dropout_rate;
        self.hyperparameters.dropout_rate = 0.0;
        let x = self.pool.new_tensor(x.to_owned(), vec![1, x.len()]);
        let y = self.call(x);

        let result: Vec<f32> = self.pool.get_data(y).to_owned();

        self.pool.flush();
        self.hyperparameters.dropout_rate = old_dropout_rate;
        result
    }

    pub fn test(&mut self, xs: &[Vec<f32>], ys: &[f32]) -> f32 {
        println!("Testing...");
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

            if y_pred as f32 == ys[i] {
                total_correct + 1.0
            } else {
                total_correct
            }
        });

        print!("\r{:50}\r", "");
        total_correct / xs.len() as f32
    }

    // —— Store ————————————————————————————————————————————————————————————————————————————
    pub fn save(&mut self) {
        let parameters = self.parameters();
        let weights: Vec<f32> = parameters
            .iter()
            .flat_map(|&p| self.pool.get_data(p).to_owned())
            .collect();
        let txt = weights
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write("model.txt", txt).unwrap();
        println!("Saved model weights");
    }

    pub fn load(&mut self) {
        let txt = fs::read_to_string("model.txt").unwrap();
        let weights: Vec<f32> = txt.lines().map(|l| l.parse().unwrap()).collect();
        let mut offset = 0;
        for &p in self.parameters().iter() {
            let size = self.pool.get_data(p).len();
            self.pool
                .set_data(p, weights[offset..offset + size].to_vec());
            offset += size;
        }
        println!("Loaded model weights");
    }
}

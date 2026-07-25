use crate::*;


// ——— Pool —————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Pool {
    nodes: Vec<TensorNode>,
    param_end: usize,
    gpu: Gpu,
}

#[derive(Copy, Clone)]
pub struct Tensor(pub usize);

impl Pool {

    // —— Constructors —————————————————————————————————————————————————————————————————————
    pub fn new() -> Self {
        let gpu = pollster::block_on(Gpu::new());
        Pool { nodes: Vec::new(), param_end: 0, gpu }
    }

    pub fn new_tensor(&mut self, data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        self.nodes.push(TensorNode::new(data, shape));
        Tensor(self.nodes.len() - 1)
    }

    pub fn new_kid(&mut self, mut node: TensorNode, parents: Vec<usize>, op: &'static str) -> Tensor {
        node.parents = parents;
        node.op = op;
        self.nodes.push(node);
        Tensor(self.nodes.len() - 1)
    }

    pub fn fill(&mut self, shape: Vec<usize>, num: f32) -> Tensor {
        self.nodes.push(TensorNode::fill(shape, num));
        Tensor(self.nodes.len() - 1)
    }

    pub fn new_rand(&mut self, shape: Vec<usize>) -> Tensor {
        self.nodes.push(TensorNode::new_rand(shape));
        Tensor(self.nodes.len() - 1)
    }

    // —— Debug ————————————————————————————————————————————————————————————————————————————
    pub fn print(&self, a: Tensor) {
        println!("Tensor(data={:?}, shape={:?})", self.nodes[a.0].data, self.nodes[a.0].shape);
    }

    // —— Optim ————————————————————————————————————————————————————————————————————————————
    pub fn set_param_end(&mut self) {
        self.param_end = self.nodes.len();
    }

    pub fn flush(&mut self) {
        self.nodes.truncate(self.param_end);
    }

    // —— Getters / setters ————————————————————————————————————————————————————————————————
    pub fn get_data(&self, t: Tensor) -> &[f32] {
        &self.nodes[t.0].data
    }

    pub fn set_data(&mut self, t: Tensor, data: Vec<f32>) {
        self.nodes[t.0].data = data;
    }

    pub fn get_shape(&self, t: Tensor) -> &[usize] {
        &self.nodes[t.0].shape
    }

    pub fn update(&mut self, t_tensor: Tensor, learning_rate: f32) {
        let t: &mut TensorNode = &mut self.nodes[t_tensor.0];
        t.data.iter_mut().zip(t.grad.iter()).for_each(|(d, g)| *d -= learning_rate * g);
    }

    // —— Forward ops ——————————————————————————————————————————————————————————————————————
    pub fn matmul(&mut self, a_tensor: Tensor, b_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let b = &self.nodes[b_tensor.0];
        let c = matmul_gpu(&self.gpu, a, b);
        self.new_kid(c, vec![a_tensor.0, b_tensor.0], "@")
    }

    pub fn bias_add(&mut self, a_tensor: Tensor, bias_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let bias = &self.nodes[bias_tensor.0];
        assert_eq!(bias.shape.len(), 1, "invalid bias node for bias add");
        assert_eq!(a.shape[1], bias.shape[0], "invalid tensors for bias add");

        let z: Vec<f32> = (0..a.shape[0]).flat_map(|row| {
            (0..bias.shape[0]).map(move |col| a.get(&[row, col]) + bias.data[col])
        }).collect();

        self.new_kid(TensorNode::new(z, a.shape.clone()), vec![a_tensor.0, bias_tensor.0], "bias +")
    }

    pub fn gather(&mut self, y_pred_tensor: Tensor, labels_tensor: Tensor) -> Tensor {
        let y_pred = &self.nodes[y_pred_tensor.0];
        let labels = &self.nodes[labels_tensor.0];

        let data = (0..labels.shape[0]).map(|batch| {
            let label = labels.get(&[batch]) as usize;
            y_pred.get(&[batch, label])
        }).collect();

        self.new_kid(TensorNode::new(data, labels.shape.clone()), vec![y_pred_tensor.0, labels_tensor.0], "gather")
    }

    pub fn log(&mut self, a_tensor: Tensor) -> Tensor {
        let data = self.nodes[a_tensor.0].data.iter().map(|v| v.ln()).collect();
        self.new_kid(TensorNode::new(data, self.nodes[a_tensor.0].shape.clone()), vec![a_tensor.0], "log")
    }

    pub fn mean(&mut self, a_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let sum: f32 = a.data.iter().sum();
        let mean = sum / a.data.len() as f32;
        self.new_kid(TensorNode::new(vec![mean], vec![1]), vec![a_tensor.0], "mean")
    }

    pub fn neg(&mut self, a_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let data = a.data.iter().map(|v| -v).collect();
        self.new_kid(TensorNode::new(data, a.shape.clone()), vec![a_tensor.0], "neg")
    }

    pub fn max(&mut self, a_tensor: Tensor, b_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let b = &self.nodes[b_tensor.0];
        assert_eq!(a.data.len(), b.data.len(), "mismatching dimensions for max");
        let data = a.data.iter().zip(b.data.iter())
            .map(|(&a_k, &b_k)| f32::max(a_k, b_k))
            .collect();
        self.new_kid(TensorNode::new(data, a.shape.clone()), vec![a_tensor.0, b_tensor.0], "max")
    }

    pub fn exp(&mut self, a_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let data = a.data.iter().map(|v| v.exp()).collect();
        self.new_kid(TensorNode::new(data, a.shape.clone()), vec![a_tensor.0], "exp")
    }

    // a = [batch, n_classes]; c = [batch]
    pub fn sum(&mut self, a_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let data = (0..a.shape[0]).map(|row| {
            (0..a.shape[1]).map(|col| a.get(&[row, col])).sum()
        }).collect();
        self.new_kid(TensorNode::new(data, vec![a.shape[0]]), vec![a_tensor.0], "sum")
    }

    // a = [batch, n_classes]; b = [batch]; c = [batch, n_classes]
    pub fn div(&mut self, a_tensor: Tensor, b_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let b = &self.nodes[b_tensor.0];
        let data = (0..a.shape[0]).flat_map(|row| {
            (0..a.shape[1]).map(move |col| a.get(&[row, col]) / b.get(&[row]))
        }).collect();
        self.new_kid(TensorNode::new(data, a.shape.clone()), vec![a_tensor.0, b_tensor.0], "div")
    }

    pub fn mul(&mut self, a_tensor: Tensor, b_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let b = &self.nodes[b_tensor.0];
        let data = a.data.iter().zip(b.data.iter())
            .map(|(a_k, b_k)| a_k * b_k)
            .collect();
        self.new_kid(TensorNode::new(data, a.shape.clone()), vec![a_tensor.0, b_tensor.0], "mul")
    }

    pub fn tanh(&mut self, a_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let data = a.data.iter().map(|a_k| {
            (1.0 - (-2.0 * a_k).exp()) / (1.0 + (-2.0 * a_k).exp())
        }).collect();
        self.new_kid(TensorNode::new(data, a.shape.clone()), vec![a_tensor.0], "tanh")
    }

    pub fn subtract_row_max(&mut self, a_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let data = (0..a.shape[0]).flat_map(|row| {
            let row_max = (0..a.shape[1]).map(|col| a.get(&[row, col])).fold(f32::NEG_INFINITY, f32::max);
            (0..a.shape[1]).map(move |col| a.get(&[row, col]) - row_max)
        }).collect();
        self.new_kid(TensorNode::new(data, a.shape.clone()), vec![a_tensor.0], "sub_row_max")
    }

    // —— Backward ops —————————————————————————————————————————————————————————————————————
    pub fn matmul_backward(&self, a: &TensorNode, b: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let at = transpose(a);
        let bt = transpose(b);
        let dc_da = matmul_gpu(&self.gpu, &dc, &bt);
        let dc_db = matmul_gpu(&self.gpu, &at, &dc);
        vec![dc_da, dc_db]
    }

    pub fn bias_add_backward(&self, dc: &TensorNode) -> Vec<TensorNode> {
        let da = TensorNode::new(dc.data.clone(), dc.shape.clone());

        let d_bias_data: Vec<f32> = (0..dc.shape[1]).map(|col| {
            (0..dc.shape[0]).map(|row| dc.get(&[row, col])).sum()
        }).collect();
        let d_bias = TensorNode::new(d_bias_data, vec![dc.shape[1]]);

        vec![da, d_bias]
    }

    pub fn gather_backward(&self, y_pred: &TensorNode, labels: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let mut dc_d_ypred = TensorNode::fill(y_pred.shape.clone(), 0.0);

        (0..y_pred.shape[0]).for_each(|batch| {
            let label = labels.get(&[batch]) as usize;
            dc_d_ypred.set(&[batch, label], dc.get(&[batch]))
        });

        let dc_dlabels = TensorNode::fill(dc.shape.clone(), 0.0);

        vec![dc_d_ypred, dc_dlabels]
    }

    pub fn log_backward(&self, a: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let data = a.data.iter().zip(dc.data.iter())
            .map(|(a_k, dc_k)| dc_k / a_k)
            .collect();
        vec![TensorNode::new(data, dc.shape.clone())]
    }

    pub fn mean_backward(&self, a: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let n = a.data.len() as f32;
        let data = a.data.iter()
            .map(|_| dc.data[0] / n)
            .collect();
        vec![TensorNode::new(data, a.shape.clone())]
    }

    pub fn neg_backward(&self, dc: &TensorNode) -> Vec<TensorNode> {
        let data = dc.data.iter()
            .map(|dc_i| -dc_i)
            .collect();
        vec![TensorNode::new(data, dc.shape.clone())]
    }

    pub fn max_backward(&self, a: &TensorNode, b: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let (data_a, data_b) = a.data.iter().zip(b.data.iter()).zip(dc.data.iter())
            .map(|((a_k, b_k), &dc_k)| if a_k > b_k { 
                (dc_k, 0.0)
            } else {
                (0.0, dc_k)
            })
        .unzip();
        vec![TensorNode::new(data_a, dc.shape.clone()), TensorNode::new(data_b, dc.shape.clone())]
    }

    pub fn exp_backward(&self, c: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let data: Vec<f32> = c.data.iter().zip(dc.data.iter())
            .map(|(&c_k, &dc_k)| c_k * dc_k)
            .collect();
        vec![TensorNode::new(data, c.shape.clone())]
    }

    pub fn broadcast_dc(&self, a: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let data: Vec<f32> = (0..a.shape[0]).flat_map(|row| {
            (0..a.shape[1]).map(move |_| dc.data[row])
        }).collect();
        vec![TensorNode::new(data, a.shape.clone())]
    }

    pub fn div_backward(&self, a: &TensorNode, b: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let mut data_a: Vec<f32> = Vec::with_capacity(a.data.len());
        let mut data_b: Vec<f32> = Vec::with_capacity(b.data.len());

        for row in 0..a.shape[0] {
            let b_k = b.get(&[row]);
            let dc_db = (0..a.shape[1]).fold(0.0, |acc, col| {
                data_a.push(dc.get(&[row, col]) / b_k);
                let a_k = a.get(&[row, col]);
                let dc_k = dc.get(&[row, col]);
                acc + a_k * dc_k
            }) * -1.0 / b_k.powi(2);
            data_b.push(dc_db);
        }

        vec![TensorNode::new(data_a, a.shape.clone()), TensorNode::new(data_b, b.shape.clone())]
    }

    pub fn mul_backward(&self, a: &TensorNode, b: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let (data_a, data_b) = a.data.iter().zip(b.data.iter()).zip(dc.data.iter())
            .map(|((a_k, b_k), dc_k)| (b_k * dc_k,  a_k * dc_k))
            .unzip();
        vec![TensorNode::new(data_a, dc.shape.clone()), TensorNode::new(data_b, dc.shape.clone())]
    }

    pub fn tanh_backward(&self, c: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let data = c.data.iter().zip(dc.data.iter())
            .map(|(c_k, dc_k)| (1.0 - c_k.powi(2)) * dc_k)
            .collect();
        vec![TensorNode::new(data, c.shape.clone())]
    }

    // —— Backpropogate ————————————————————————————————————————————————————————————————————
    pub fn backpropogate(&mut self, root_tensor: Tensor) {

        // Zero grads
        (0..root_tensor.0).for_each(|i| self.nodes[i].set_grad(0.0));

        // Initial node
        let root = &mut self.nodes[root_tensor.0];
        root.set_grad(1.0);

        // Run backward
        for i in (0..=root_tensor.0).rev() {
            let cur = &self.nodes[i];
            if cur.parents.len() == 0 { continue; }

            let cur_grad = TensorNode::new(cur.grad.clone(), cur.shape.clone());
            let par_1 = &self.nodes[cur.parents[0]];
            let par_2 = if cur.parents.len() == 2 { Some(&self.nodes[cur.parents[1]]) } else { None };

            let par_grads = match self.nodes[i].op {
                "@" => self.matmul_backward(par_1, par_2.unwrap(), &cur_grad),
                "bias +" => self.bias_add_backward(&cur_grad),
                "gather" => self.gather_backward(par_1, par_2.unwrap(), &cur_grad),
                "log" => self.log_backward(par_1, &cur_grad),
                "mean" => self.mean_backward(par_1, &cur_grad),
                "neg" => self.neg_backward(&cur_grad),
                "max" => self.max_backward(par_1, par_2.unwrap(), &cur_grad),
                "exp" => self.exp_backward(cur, &cur_grad),
                "sum" => self.broadcast_dc(par_1, &cur_grad),
                "div" => self.div_backward(par_1, par_2.unwrap(), &cur_grad),
                "mul" => self.mul_backward(par_1, par_2.unwrap(), &cur_grad),
                "tanh" => self.tanh_backward(&cur, &cur_grad),
                "sub_row_max" => self.broadcast_dc(par_1, &cur_grad),
                op => { panic!("{} not accounted for", op) }
            };

            for p in 0..cur.parents.len() {
                let par_idx = self.nodes[i].parents[p];
                let old_par_grad = TensorNode::new(self.nodes[par_idx].grad.clone(), self.nodes[par_idx].shape.clone());
                self.nodes[par_idx].grad = matadd(&old_par_grad, &par_grads[p]).data;
            }
        }
    }
}


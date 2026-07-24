use crate::*;


// ——— TensorPool —————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct TensorPool {
    nodes: Vec<TensorNode>,
}

#[derive(Copy, Clone)]
pub struct Tensor(pub usize);

impl TensorPool {

    // —— Constructors —————————————————————————————————————————————————————————————————————
    pub fn new() -> Self {
        TensorPool { nodes: Vec::new() }
    }

    pub fn new_tensor(&mut self, data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        self.nodes.push(TensorNode::new(data, shape));
        Tensor(self.nodes.len() - 1)
    }

    pub fn new_kid(&mut self, mut node: TensorNode, parents: Vec<usize>, op: &'static str) -> Tensor {
        node.parents = parents;
        node.op = op;
        self.nodes.push(node);
        Tensor(self.nodes.len() - 1)
    }

    pub fn new_zeros(&mut self, shape: Vec<usize>) -> Tensor {
        self.nodes.push(TensorNode::new_zeros(shape));
        Tensor(self.nodes.len() - 1)
    }

    pub fn new_rand(&mut self, shape: Vec<usize>) -> Tensor {
        self.nodes.push(TensorNode::new_rand(shape));
        Tensor(self.nodes.len() - 1)
    }

    // —— Debug ————————————————————————————————————————————————————————————————————————————
    pub fn print(&self, a: Tensor) {
        println!("Value(data={:?})", self.nodes[a.0].data);
    }

    // —— Getters / setters ————————————————————————————————————————————————————————————————
    pub fn get_data(&self, t: Tensor) -> &[f64] {
        &self.nodes[t.0].data
    }

    pub fn set_data(&mut self, t: Tensor, data: Vec<f64>) {
        self.nodes[t.0].data = data;
    }

    pub fn update(&mut self, t_tensor: Tensor, learning_rate: f64) {
        let t: &mut TensorNode = &mut self.nodes[t_tensor.0];
        t.data.iter_mut().zip(t.grad.iter()).for_each(|(d, g)| *d -= learning_rate * g);
    }

    // —— Forward ops ——————————————————————————————————————————————————————————————————————
    pub fn matmul(&mut self, a_tensor: Tensor, b_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let b = &self.nodes[b_tensor.0];
        let c = matmul(a, b);
        self.new_kid(c, vec![a_tensor.0, b_tensor.0], "@")
    }

    pub fn bias_add(&mut self, a_tensor: Tensor, bias_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let bias = &self.nodes[bias_tensor.0];
        assert_eq!(bias.shape.len(), 1, "invalid bias node for bias add");
        assert_eq!(a.shape[1], bias.shape[0], "invalid tensors for bias add");

        let z: Vec<f64> = (0..a.shape[0]).flat_map(|row| {
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

    pub fn log(&mut self, a: Tensor) -> Tensor {
        let data = self.nodes[a.0].data.iter().map(|v| v.ln()).collect();
        self.new_kid(TensorNode::new(data, self.nodes[a.0].shape.clone()), vec![a.0], "log")
    }

    pub fn mean(&mut self, a: Tensor) -> Tensor {


        todo!()
    }

    pub fn neg(&mut self, a: Tensor) -> Tensor {
        todo!()
    }

    // —— Backward ops —————————————————————————————————————————————————————————————————————
    pub fn matmul_backward(&self, a: &TensorNode, b: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let at = transpose(a);
        let bt = transpose(b);
        let dc_da = matmul(&dc, &bt);
        let dc_db = matmul(&at, &dc);
        vec!(dc_da, dc_db)
    }

    pub fn bias_add_backward(&self, dc: &TensorNode) -> Vec<TensorNode> {
        let da = TensorNode::new(dc.data.clone(), dc.shape.clone());

        let d_bias_data: Vec<f64> = (0..dc.shape[1]).map(|col| {
            (0..dc.shape[0]).map(|row| dc.get(&[row, col])).sum()
        }).collect();
        let d_bias = TensorNode::new(d_bias_data, vec![dc.shape[1]]);

        vec!(da, d_bias)
    }

    pub fn gather_backward(&self, y_pred: &TensorNode, labels: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let mut dc_d_ypred = TensorNode::new_zeros(y_pred.shape.clone());

        (0..y_pred.shape[0]).for_each(|batch| {
            let label = labels.get(&[batch]) as usize;
            dc_d_ypred.set(&[batch, label], dc.get(&[batch]))
        });

        let dc_dlabels = TensorNode::new_zeros(dc.shape.clone());

        vec!(dc_d_ypred, dc_dlabels)
    }

    pub fn log_backward(&self, a: &TensorNode, dc: &TensorNode) -> Vec<TensorNode> {
        let data = a.data.iter().zip(dc.data.iter())
            .map(|(a_k, dc_k)| dc_k / a_k)
            .collect();
        vec!(TensorNode::new(data, dc.shape.clone()))
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
                "gather" => self.gather_backward(&par_1, &par_2.unwrap(), &cur_grad),
                "log" => self.log_backward(&par_1, &cur_grad),
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


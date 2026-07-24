use crate::*;


// ——— TensorPool —————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct TensorPool {
    pub nodes: Vec<TensorNode>,
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

    // —— Backward ops —————————————————————————————————————————————————————————————————————
    pub fn matmul_backward(&self, a: &TensorNode, b: &TensorNode, dc: &TensorNode) -> (TensorNode, TensorNode) {
        let at = transpose(a);
        let bt = transpose(b);
        let dc_da = matmul(&dc, &bt);
        let dc_db = matmul(&at, &dc);
        (dc_da, dc_db)
    }

    pub fn bias_add_backward(&self, dc: &TensorNode) -> (TensorNode, TensorNode) {
        let da = TensorNode::new(dc.data.clone(), dc.shape.clone());

        let d_bias_data: Vec<f64> = (0..dc.shape[1]).map(|col| {
            (0..dc.shape[0]).map(|row| dc.get(&[row, col])).sum()
        }).collect();
        let d_bias = TensorNode::new(d_bias_data, vec![dc.shape[1]]);

        (da, d_bias)
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
            let cur_grad = TensorNode::new(cur.grad.clone(), cur.shape.clone());
            if cur.parents.len() == 0 { continue; }

            let par_grads = match self.nodes[i].op {
                "@" => {
                    let par_1 = &self.nodes[cur.parents[0]];
                    let par_2 = &self.nodes[cur.parents[1]];
                    self.matmul_backward(par_1, par_2, &cur_grad)
                },
                "bias +" => self.bias_add_backward(&cur_grad),
                op => { panic!("{} not accounted for", op) }
            };

            for p in 0..cur.parents.len() {
                let par_idx = self.nodes[i].parents[p];
                let old_par_grad = TensorNode::new(self.nodes[par_idx].grad.clone(), self.nodes[par_idx].shape.clone());
                let par_grad = match p { 0 => &par_grads.0, 1 => &par_grads.1, _ => panic!() };
                self.nodes[par_idx].grad = matadd(&old_par_grad, par_grad).data;
            }
        }
    }
}


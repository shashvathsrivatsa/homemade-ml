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

    // —— Forward ops ——————————————————————————————————————————————————————————————————————
    pub fn transpose(&mut self, a_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let a_t = transpose(a);
        self.new_kid(a_t, vec![a_tensor.0], "T")
    }

    pub fn matmul(&mut self, a_tensor: Tensor, b_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let b = &self.nodes[b_tensor.0];
        let c = matmul(a, b);
        self.new_kid(c, vec![a_tensor.0, b_tensor.0], "@")
    }

    pub fn matadd(&mut self, a_tensor: Tensor, b_tensor: Tensor) -> Tensor {
        let a = &self.nodes[a_tensor.0];
        let b = &self.nodes[b_tensor.0];
        let c = matadd(a, b);
        self.new_kid(c, vec![a_tensor.0, b_tensor.0], "+")
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
    pub fn matmul_backward(&mut self, a_tensor: Tensor, b_tensor: Tensor, dc_tensor: Tensor) -> (TensorNode, TensorNode) {
        let a = &self.nodes[a_tensor.0];
        let b = &self.nodes[b_tensor.0];
        let dc = &self.nodes[dc_tensor.0];

        let at = transpose(a);
        let bt = transpose(b);
        let dc_da = matmul(&dc, &bt);
        let dc_db = matmul(&at, &dc);
        (dc_da, dc_db)
    }

    pub fn bias_add_backward(&mut self, ) {
        todo!()
    }
}


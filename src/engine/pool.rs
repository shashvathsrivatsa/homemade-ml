use crate::*;


// ——— Pool ———————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Pool {
    nodes: Vec<TensorNode>,
    param_end: usize,
    gpu: Gpu,
}

#[derive(Copy, Clone)]
pub struct Tensor(pub usize);

impl Pool {

    pub fn new() -> Self {
        Self {
            nodes: vec![],
            param_end: 0,
            gpu: pollster::block_on(Gpu::new()),
        }
    }

    fn node_from_buffer(&self, data: wgpu::Buffer, shape: Vec<usize>) -> TensorNode {
        let len = shape.iter().product();
        let grad = self.gpu.empty_buffer(len);
        TensorNode::from_buffers(data, grad, shape)
    }

    fn push_node(&mut self, mut node: TensorNode, parents: Vec<usize>, op: &'static str) -> Tensor {
        node.parents = parents;
        node.op = op;
        self.nodes.push(node);
        Tensor(self.nodes.len() - 1)
    }

    pub fn upload(&mut self, data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "data does not match tensor shape"
        );
        let buffer = self.gpu.upload_buffer(&data);
        let node = self.node_from_buffer(buffer, shape);
        self.push_node(node, vec![], "")
    }

    pub fn new_tensor(&mut self, data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        self.upload(data, shape)
    }

    pub fn fill(&mut self, shape: Vec<usize>, num: f32) -> Tensor {
        self.upload(vec![num; shape.iter().product()], shape)
    }

    pub fn new_rand(&mut self, shape: Vec<usize>) -> Tensor {
        let mut rng = thread_rng();
        let data = (0..shape.iter().product())
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        self.upload(data, shape)
    }

    pub fn download(&self, t: Tensor) -> Vec<f32> {
        let node = &self.nodes[t.0];
        self.gpu.download_buffer(&node.data, node.len)
    }

    pub fn download_grad(&self, t: Tensor) -> Vec<f32> {
        let node = &self.nodes[t.0];
        self.gpu.download_buffer(&node.grad, node.len)
    }

    pub fn get_data(&self, t: Tensor) -> Vec<f32> {
        self.download(t)
    }
    pub fn get_grad(&self, t: Tensor) -> Vec<f32> {
        self.download_grad(t)
    }
    pub fn get_shape(&self, t: Tensor) -> &[usize] {
        &self.nodes[t.0].shape
    }

    pub fn set_data(&mut self, t: Tensor, data: Vec<f32>) {
        assert_eq!(data.len(), self.nodes[t.0].len);
        self.gpu
            .queue
            .write_buffer(&self.nodes[t.0].data, 0, bytemuck::cast_slice(&data));
    }

    pub fn print(&self, t: Tensor) {
        println!(
            "Tensor(data={:?}, shape={:?})",
            self.download(t),
            self.nodes[t.0].shape
        );
    }

    pub fn set_param_end(&mut self) {
        self.param_end = self.nodes.len();
    }
    pub fn flush(&mut self) {
        self.nodes.truncate(self.param_end);
    }

    fn groups_1d(n: usize) -> (u32, u32, u32) {
        (n.div_ceil(256) as u32, 1, 1)
    }

    fn unary_buffer(&self, a: &wgpu::Buffer, len: usize, kind: u32) -> wgpu::Buffer {
        let out = self.gpu.empty_buffer(len);
        self.gpu.dispatch(
            "unary",
            &[a, a],
            &[&out],
            [len as u32, kind, 0, 0],
            Self::groups_1d(len),
        );
        out
    }

    fn binary_buffer(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        len: usize,
        kind: u32,
    ) -> wgpu::Buffer {
        let out = self.gpu.empty_buffer(len);
        self.gpu.dispatch(
            "binary",
            &[a, b],
            &[&out],
            [len as u32, kind, 0, 0],
            Self::groups_1d(len),
        );
        out
    }

    fn unary_op(&mut self, a: Tensor, kind: u32, op: &'static str) -> Tensor {
        let shape = self.nodes[a.0].shape.clone();
        let data = self.unary_buffer(&self.nodes[a.0].data, self.nodes[a.0].len, kind);
        let node = self.node_from_buffer(data, shape);
        self.push_node(node, vec![a.0], op)
    }

    fn binary_op(&mut self, a: Tensor, b: Tensor, kind: u32, op: &'static str) -> Tensor {
        assert_eq!(
            self.nodes[a.0].shape, self.nodes[b.0].shape,
            "mismatching dimensions"
        );
        let shape = self.nodes[a.0].shape.clone();
        let data = self.binary_buffer(
            &self.nodes[a.0].data,
            &self.nodes[b.0].data,
            self.nodes[a.0].len,
            kind,
        );
        let node = self.node_from_buffer(data, shape);
        self.push_node(node, vec![a.0, b.0], op)
    }

    pub fn matmul(&mut self, a: Tensor, b: Tensor) -> Tensor {
        let (m, k) = (self.nodes[a.0].shape[0], self.nodes[a.0].shape[1]);
        let n = self.nodes[b.0].shape[1];
        assert_eq!(k, self.nodes[b.0].shape[0], "invalid dimensions for matmul");
        let data = self.gpu.empty_buffer(m * n);
        self.gpu.dispatch(
            "matmul",
            &[&self.nodes[a.0].data, &self.nodes[b.0].data],
            &[&data],
            [m as u32, k as u32, n as u32, 0],
            (m.div_ceil(16) as u32, n.div_ceil(16) as u32, 1),
        );
        let node = self.node_from_buffer(data, vec![m, n]);
        self.push_node(node, vec![a.0, b.0], "@")
    }

    pub fn bias_add(&mut self, a: Tensor, bias: Tensor) -> Tensor {
        let (rows, cols) = (self.nodes[a.0].shape[0], self.nodes[a.0].shape[1]);
        assert_eq!(
            self.nodes[bias.0].shape,
            vec![cols],
            "invalid bias node for bias add"
        );
        let data = self.gpu.empty_buffer(rows * cols);
        self.gpu.dispatch(
            "bias_add",
            &[&self.nodes[a.0].data, &self.nodes[bias.0].data],
            &[&data],
            [rows as u32, cols as u32, 0, 0],
            Self::groups_1d(rows * cols),
        );
        let node = self.node_from_buffer(data, vec![rows, cols]);
        self.push_node(node, vec![a.0, bias.0], "bias +")
    }

    pub fn gather(&mut self, values: Tensor, labels: Tensor) -> Tensor {
        let rows = self.nodes[labels.0].len;
        let cols = self.nodes[values.0].shape[1];
        let data = self.gpu.empty_buffer(rows);
        self.gpu.dispatch(
            "gather",
            &[&self.nodes[values.0].data, &self.nodes[labels.0].data],
            &[&data],
            [rows as u32, cols as u32, 0, 0],
            Self::groups_1d(rows),
        );
        let node = self.node_from_buffer(data, vec![rows]);
        self.push_node(node, vec![values.0, labels.0], "gather")
    }

    pub fn log(&mut self, a: Tensor) -> Tensor {
        self.unary_op(a, 2, "log")
    }
    pub fn neg(&mut self, a: Tensor) -> Tensor {
        self.unary_op(a, 3, "neg")
    }
    pub fn exp(&mut self, a: Tensor) -> Tensor {
        self.unary_op(a, 1, "exp")
    }
    pub fn tanh(&mut self, a: Tensor) -> Tensor {
        self.unary_op(a, 0, "tanh")
    }
    pub fn mul(&mut self, a: Tensor, b: Tensor) -> Tensor {
        self.binary_op(a, b, 1, "mul")
    }
    pub fn max(&mut self, a: Tensor, b: Tensor) -> Tensor {
        self.binary_op(a, b, 2, "max")
    }
    pub fn matadd(&mut self, a: Tensor, b: Tensor) -> Tensor {
        self.binary_op(a, b, 0, "matadd")
    }

    pub fn mean(&mut self, a: Tensor) -> Tensor {
        let len = self.nodes[a.0].len;
        let sum = self.gpu.empty_buffer(1);
        self.gpu.dispatch(
            "sum",
            &[&self.nodes[a.0].data, &self.nodes[a.0].data],
            &[&sum],
            [1, len as u32, 0, 0],
            (1, 1, 1),
        );
        let divisor = self.gpu.upload_buffer(&[len as f32]);
        let data = self.gpu.empty_buffer(1);
        self.gpu
            .dispatch("div", &[&sum, &divisor], &[&data], [1, 1, 0, 0], (1, 1, 1));
        let node = self.node_from_buffer(data, vec![1]);
        self.push_node(node, vec![a.0], "mean")
    }

    pub fn sum(&mut self, a: Tensor) -> Tensor {
        let (rows, cols) = (self.nodes[a.0].shape[0], self.nodes[a.0].shape[1]);
        let data = self.gpu.empty_buffer(rows);
        self.gpu.dispatch(
            "sum",
            &[&self.nodes[a.0].data, &self.nodes[a.0].data],
            &[&data],
            [rows as u32, cols as u32, 0, 0],
            Self::groups_1d(rows),
        );
        let node = self.node_from_buffer(data, vec![rows]);
        self.push_node(node, vec![a.0], "sum")
    }

    pub fn div(&mut self, a: Tensor, b: Tensor) -> Tensor {
        let (rows, cols) = (self.nodes[a.0].shape[0], self.nodes[a.0].shape[1]);
        assert_eq!(self.nodes[b.0].len, rows);
        let data = self.gpu.empty_buffer(rows * cols);
        self.gpu.dispatch(
            "div",
            &[&self.nodes[a.0].data, &self.nodes[b.0].data],
            &[&data],
            [rows as u32, cols as u32, 0, 0],
            Self::groups_1d(rows * cols),
        );
        let node = self.node_from_buffer(data, self.nodes[a.0].shape.clone());
        self.push_node(node, vec![a.0, b.0], "div")
    }

    pub fn sub_row_max(&mut self, a: Tensor) -> Tensor {
        let (rows, cols) = (self.nodes[a.0].shape[0], self.nodes[a.0].shape[1]);
        let data = self.gpu.empty_buffer(rows * cols);
        self.gpu.dispatch(
            "sub_row_max",
            &[&self.nodes[a.0].data, &self.nodes[a.0].data],
            &[&data],
            [rows as u32, cols as u32, 0, 0],
            Self::groups_1d(rows),
        );
        let node = self.node_from_buffer(data, self.nodes[a.0].shape.clone());
        self.push_node(node, vec![a.0], "sub_row_max")
    }

    pub fn update(&mut self, t: Tensor, learning_rate: f32) {
        let node = &self.nodes[t.0];
        let out = self.gpu.empty_buffer(node.len);
        self.gpu.dispatch(
            "update",
            &[&node.data, &node.grad],
            &[&out],
            [node.len as u32, learning_rate.to_bits(), 0, 0],
            Self::groups_1d(node.len),
        );
        self.nodes[t.0].data = out;
    }

    fn backward_for(&self, i: usize) -> Vec<wgpu::Buffer> {
        let cur = &self.nodes[i];
        let a = &self.nodes[cur.parents[0]];
        let b = cur.parents.get(1).map(|&p| &self.nodes[p]);
        let alloc = |len| self.gpu.empty_buffer(len);
        match cur.op {
            "@" => {
                let b = b.unwrap();
                let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
                let da = alloc(a.len);
                self.gpu.dispatch(
                    "matmul",
                    &[&cur.grad, &b.data],
                    &[&da],
                    [m as u32, n as u32, k as u32, 2],
                    (m.div_ceil(16) as u32, k.div_ceil(16) as u32, 1),
                );
                let db = alloc(b.len);
                self.gpu.dispatch(
                    "matmul",
                    &[&a.data, &cur.grad],
                    &[&db],
                    [k as u32, m as u32, n as u32, 1],
                    (k.div_ceil(16) as u32, n.div_ceil(16) as u32, 1),
                );
                vec![da, db]
            }
            "bias +" => {
                let da = self.unary_buffer(&cur.grad, cur.len, 9);
                let db = alloc(b.unwrap().len);
                self.gpu.dispatch(
                    "bias_add_backward",
                    &[&cur.grad, &cur.grad],
                    &[&db],
                    [cur.shape[0] as u32, cur.shape[1] as u32, 0, 0],
                    Self::groups_1d(cur.shape[1]),
                );
                vec![da, db]
            }
            "gather" => {
                let labels = b.unwrap();
                let da = alloc(a.len);
                self.gpu.dispatch(
                    "gather_backward",
                    &[&cur.grad, &labels.data],
                    &[&da],
                    [a.shape[0] as u32, a.shape[1] as u32, 0, 0],
                    Self::groups_1d(a.len),
                );
                vec![da, self.gpu.empty_buffer(labels.len)]
            }
            "log" => {
                let out = alloc(a.len);
                self.gpu.dispatch(
                    "unary",
                    &[&a.data, &cur.grad],
                    &[&out],
                    [a.len as u32, 6, 0, 0],
                    Self::groups_1d(a.len),
                );
                vec![out]
            }
            "mean" => {
                let out = alloc(a.len);
                self.gpu.dispatch(
                    "mean_backward",
                    &[&cur.grad, &cur.grad],
                    &[&out],
                    [a.len as u32, 0, 0, 0],
                    Self::groups_1d(a.len),
                );
                vec![out]
            }
            "neg" => vec![self.unary_buffer(&cur.grad, cur.len, 7)],
            "exp" | "tanh" => {
                let out = alloc(cur.len);
                let kind = if cur.op == "exp" { 5 } else { 8 };
                self.gpu.dispatch(
                    "unary",
                    &[&cur.data, &cur.grad],
                    &[&out],
                    [cur.len as u32, kind, 0, 0],
                    Self::groups_1d(cur.len),
                );
                vec![out]
            }
            "max" | "mul" => {
                let b = b.unwrap();
                let kinds = if cur.op == "max" { (5, 6) } else { (3, 4) };
                let da = alloc(cur.len);
                let db = alloc(cur.len);
                self.gpu.copy_buffer(&cur.grad, &da, cur.len);
                self.gpu.copy_buffer(&cur.grad, &db, cur.len);
                self.gpu.dispatch(
                    "binary",
                    &[&a.data, &b.data],
                    &[&da],
                    [cur.len as u32, kinds.0, 0, 0],
                    Self::groups_1d(cur.len),
                );
                self.gpu.dispatch(
                    "binary",
                    &[&a.data, &b.data],
                    &[&db],
                    [cur.len as u32, kinds.1, 0, 0],
                    Self::groups_1d(cur.len),
                );
                vec![da, db]
            }
            "sum" => {
                let out = alloc(a.len);
                self.gpu.dispatch(
                    "sum_backward",
                    &[&cur.grad, &cur.grad],
                    &[&out],
                    [a.shape[0] as u32, a.shape[1] as u32, 0, 0],
                    Self::groups_1d(a.len),
                );
                vec![out]
            }
            "div" => {
                let b = b.unwrap();
                let da = alloc(a.len);
                self.gpu.dispatch(
                    "div_backward_a",
                    &[&b.data, &cur.grad],
                    &[&da],
                    [a.shape[0] as u32, a.shape[1] as u32, 0, 0],
                    Self::groups_1d(a.len),
                );
                let db = alloc(a.len);
                self.gpu.copy_buffer(&cur.grad, &db, cur.len);
                self.gpu.dispatch(
                    "div_backward_b",
                    &[&a.data, &b.data],
                    &[&db],
                    [a.shape[0] as u32, a.shape[1] as u32, 0, 0],
                    Self::groups_1d(a.shape[0]),
                );
                vec![da, db]
            }
            "sub_row_max" => {
                let out = alloc(a.len);
                self.gpu.dispatch(
                    "sub_row_max_backward",
                    &[&a.data, &cur.grad],
                    &[&out],
                    [a.shape[0] as u32, a.shape[1] as u32, 0, 0],
                    Self::groups_1d(a.shape[0]),
                );
                vec![out]
            }
            "matadd" => vec![
                self.unary_buffer(&cur.grad, cur.len, 9),
                self.unary_buffer(&cur.grad, cur.len, 9),
            ],
            op => panic!("{op} not accounted for"),
        }
    }

    pub fn backpropogate(&mut self, root: Tensor) {
        for node in &self.nodes[..=root.0] {
            self.gpu.clear_buffer(&node.grad);
        }
        self.gpu.dispatch(
            "unary",
            &[&self.gpu.one, &self.gpu.one],
            &[&self.nodes[root.0].grad],
            [self.nodes[root.0].len as u32, 10, 0, 0],
            Self::groups_1d(self.nodes[root.0].len),
        );
        for i in (0..=root.0).rev() {
            if self.nodes[i].parents.is_empty() {
                continue;
            }
            let grads = self.backward_for(i);
            for (parent_slot, contribution) in grads.into_iter().enumerate() {
                let parent = self.nodes[i].parents[parent_slot];
                let accumulated = self.binary_buffer(
                    &self.nodes[parent].grad,
                    &contribution,
                    self.nodes[parent].len,
                    0,
                );
                self.nodes[parent].grad = accumulated;
            }
        }
    }
}

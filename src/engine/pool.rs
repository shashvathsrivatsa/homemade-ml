use crate::*;

// ——— Pool ———————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Pool {
    pub nodes: Vec<TensorNode>,
    param_end: usize,
    pub gpu: Gpu,
}

#[derive(Copy, Clone)]
pub struct Tensor(pub usize);

impl Pool {

    // —— Constructors —————————————————————————————————————————————————————————————————————
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            param_end: 0,
            gpu: Gpu::new(),
        }
    }

    pub fn node_from_buffer(&self, data: wgpu::Buffer, shape: Vec<usize>) -> TensorNode {
        let len = shape.iter().product();
        let grad = self.gpu.empty_buffer(len);
        TensorNode::from_buffers(data, grad, shape)
    }

    pub fn push_node(&mut self, mut node: TensorNode, parents: Vec<usize>, op: &'static str) -> Tensor {
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

    // —— Getters / setters ————————————————————————————————————————————————————————————————
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
    pub fn get_data_2d(&self, t: Tensor) -> Vec<Vec<f32>> {
        let cols = self.nodes[t.0].shape[1];
        self.download(t).chunks(cols).map(|c| c.to_vec()).collect()
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

    // —— Debug ————————————————————————————————————————————————————————————————————————————
    pub fn print(&self, t: Tensor) {
        println!(
            "Tensor(data={:?}, shape={:?})",
            self.download(t),
            self.nodes[t.0].shape
        );
    }

    // —— Optim ————————————————————————————————————————————————————————————————————————————
    pub fn set_param_end(&mut self) {
        self.param_end = self.nodes.len();
    }
    pub fn flush(&mut self) {
        self.nodes.truncate(self.param_end);
    }

    // —— Update ———————————————————————————————————————————————————————————————————————————
    pub fn update_weights(&mut self, opt: &mut OptimizerData) {
        for i in 0..self.param_end {
            match opt {
                OptimizerData::SGD(sgd) => {
                    let node = &mut self.nodes[i];
                    sgd.step(&self.gpu, node);
                }
                OptimizerData::Adam(opts) => {
                    let node = &mut self.nodes[i];
                    let adam = &mut opts[i];
                    adam.step(&self.gpu, node);
                }
            }
        }
    }

    // —— Backpropogate ————————————————————————————————————————————————————————————————————
    pub fn backpropogate(&mut self, root: Tensor) {

        // Zero grads
        for node in &self.nodes[..=root.0] {
            self.gpu.clear_buffer(&node.grad);
        }

        // Initial node
        self.gpu.dispatch(
            "unary",
            &[&self.gpu.one, &self.gpu.one],
            &[&self.nodes[root.0].grad],
            [self.nodes[root.0].len as u32, 10, 0, 0],
            Self::groups_1d(self.nodes[root.0].len),
        );

        // Run backward
        for i in (0..=root.0).rev() {
            if self.nodes[i].parents.is_empty() { continue; }
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

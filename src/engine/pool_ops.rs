use crate::*;

// ——— Pool Ops ———————————————————————————————————————————————————————————————————————————————————————————————————————

impl Pool {

    // —— Helpers ——————————————————————————————————————————————————————————————————————————

    pub fn groups_1d(n: usize) -> (u32, u32, u32) {
        const WG: usize = 256;
        let total = n.div_ceil(WG) as u32;
        let x = total.min(65535);
        let y = (total + 65534) / 65535;
        (x, y, 1)
    }

    pub fn unary_buffer(&self, a: &wgpu::Buffer, len: usize, kind: u32) -> wgpu::Buffer {
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

    pub fn binary_buffer(
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

    // —— Forward ops ——————————————————————————————————————————————————————————————————————

    pub fn reshape(&mut self, a: Tensor, shape: Vec<usize>) -> Tensor {
        assert_eq!(
            self.nodes[a.0].len,
            shape.iter().product(),
            "cannot reshape tensor with different element count"
        );
        let data = self.unary_buffer(&self.nodes[a.0].data, self.nodes[a.0].len, 9);
        let node = self.node_from_buffer(data, shape);
        self.push_node(node, vec![a.0], "reshape")
    }

    pub fn matmul(&mut self, a: Tensor, b: Tensor) -> Tensor {
        let a_squeezed = self.nodes[a.0].shape.len() == 1;
        let a = if a_squeezed {
            let n = self.nodes[a.0].shape[0];
            self.reshape(a, vec![1, n])
        } else {
            a
        };

        let b_squeezed = self.nodes[b.0].shape.len() == 1;
        let b = if b_squeezed {
            let n = self.nodes[b.0].shape[0];
            self.reshape(b, vec![n, 1])
        } else {
            b
        };

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
    pub fn sq(&mut self, a: Tensor) -> Tensor {
        self.unary_op(a, 12, "sq")
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
    pub fn sub(&mut self, a: Tensor, b: Tensor) -> Tensor {
        self.binary_op(a, b, 7, "sub")
    }

    pub fn mean(&mut self, a: Tensor) -> Tensor {
        if self.nodes[a.0].shape.len() == 1 {
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
            self.gpu.dispatch("div", &[&sum, &divisor], &[&data], [1, 1, 0, 0], (1, 1, 1));
            let node = self.node_from_buffer(data, vec![1]);
            self.push_node(node, vec![a.0], "mean")
        } else {
            let (rows, cols) = (self.nodes[a.0].shape[0], self.nodes[a.0].shape[1]);
            let data = self.gpu.empty_buffer(rows);
            self.gpu.dispatch(
                "row_mean",
                &[&self.nodes[a.0].data, &self.nodes[a.0].data],
                &[&data],
                [rows as u32, cols as u32, 0, 0],
                Self::groups_1d(rows),
            );
            let node = self.node_from_buffer(data, vec![rows]);
            self.push_node(node, vec![a.0], "mean")
        }
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

    pub fn dropout(&mut self, a: Tensor, rate: f32) -> Tensor {
        assert!(
            rate.is_finite() && (0.0..1.0).contains(&rate),
            "dropout rate must be finite and in [0, 1)"
        );
        let seed = rand::random::<u32>();
        let shape = self.nodes[a.0].shape.clone();
        let len = self.nodes[a.0].len;
        let out = self.gpu.empty_buffer(len);
        let mask = self.gpu.empty_buffer(len);
        self.gpu.dispatch(
            "dropout",
            &[&self.nodes[a.0].data],
            &[&out, &mask],
            [len as u32, rate.to_bits(), seed, 0],
            Self::groups_1d(len),
        );
        let mut node = self.node_from_buffer(out, shape);
        node.saved.push(mask);
        self.push_node(node, vec![a.0], "dropout")
    }


    // —— Backward ops —————————————————————————————————————————————————————————————————————

    pub fn backward_for(&self, i: usize) -> Vec<wgpu::Buffer> {
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
                if a.shape.len() == 1 {
                    self.gpu.dispatch(
                        "mean_backward",
                        &[&cur.grad, &cur.grad],
                        &[&out],
                        [a.len as u32, 0, 0, 0],
                        Self::groups_1d(a.len),
                    );
                } else {
                    self.gpu.dispatch(
                        "row_mean_backward",
                        &[&cur.grad, &cur.grad],
                        &[&out],
                        [a.shape[0] as u32, a.shape[1] as u32, 0, 0],
                        Self::groups_1d(a.len),
                    );
                }
                vec![out]
            }
            "neg" => vec![self.unary_buffer(&cur.grad, cur.len, 7)],
            "sq" => {
                let out = alloc(a.len);
                self.gpu.dispatch(
                    "unary",
                    &[&a.data, &cur.grad],
                    &[&out],
                    [a.len as u32, 13, 0, 0],
                    Self::groups_1d(a.len),
                );
                vec![out]
            }
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
            "sub" => vec![
                self.unary_buffer(&cur.grad, cur.len, 9),
                self.unary_buffer(&cur.grad, cur.len, 3),
            ],
            "reshape" => vec![self.unary_buffer(&cur.grad, cur.len, 9)],
            "dropout" => {
                let mask = &cur.saved[0];
                vec![self.binary_buffer(mask, &cur.grad, cur.len, 1)]
            }
            op => panic!("{op} not accounted for"),
        }
    }
}

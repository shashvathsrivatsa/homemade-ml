// use crate::*;


// ——— Value ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Pool {
    pub nodes: Vec<Node>
}

pub struct Node {
    pub data: f64,
    pub grad: f64,
    pub parents: Vec<usize>,
    pub op: &'static str,
}

#[derive(Copy, Clone)]
pub struct Value(pub usize);

impl Pool {
    pub fn new() -> Self {
        Pool { nodes: Vec::new() }
    }

    // —— Value constructors ———————————————————————————————————————————————————————————————
    pub fn new_value(&mut self, data: f64) -> Value {
        self.nodes.push(Node { data, grad: 0.0, parents: vec![], op: "" });
        Value(self.nodes.len() - 1)
    }

    pub fn new_kid(&mut self, data: f64, parents: Vec<usize>, op: &'static str) -> Value {
        self.nodes.push(Node { data, grad: 0.0, parents, op });
        Value(self.nodes.len() - 1)
    }

    // —— Debug ————————————————————————————————————————————————————————————————————————————
    pub fn print(&self, value: Value) {
        println!("Value(data={})", self.nodes[value.0].data);
    }

    // —— Setters / getters ————————————————————————————————————————————————————————————————
    pub fn set_data(&mut self, v: Value, data: f64) {
        self.nodes[v.0].data = data;
    }

    pub fn get_data(&self, v: Value) -> f64 {
        self.nodes[v.0].data
    }

    pub fn set_grad(&mut self, v: Value, grad: f64) {
        self.nodes[v.0].grad = grad;
    }

    pub fn get_grad(&self, v: Value) -> f64 {
        self.nodes[v.0].grad
    }

    pub fn update(&mut self, v: Value, learning_rate: f64) {
        let node: &mut Node = &mut self.nodes[v.0];
        node.data -= learning_rate * node.grad;
    }

    // —— Backpropogate ————————————————————————————————————————————————————————————————————
    pub fn backpropogate(&mut self, root: Value) {

        // Zero grads
        self.nodes.iter_mut().for_each(|n| n.grad = 0.0);

        // Initial node
        self.set_grad(root, 1.0);

        // Run backwards through each node and feed both of its parents
        for i in (0..=root.0).rev() {
            for p in 0..self.nodes[i].parents.len() {
                let grad = match self.nodes[i].op {
                    "+"    => self.nodes[i].grad,
                    "*"    => self.nodes[self.nodes[i].parents[(p+1) % 2]].data * self.nodes[i].grad,
                    "-"    => if p == 0 { self.nodes[i].grad } else { -self.nodes[i].grad },
                    "pow2" => 2.0 * self.nodes[self.nodes[i].parents[0]].data * self.nodes[i].grad,
                    "tanh" => (1.0 - self.nodes[i].data.powi(2)) * self.nodes[i].grad,
                    op => { println!("{} not accounted for", op); 0.0 }
                };

                let parent_idx = self.nodes[i].parents[p];
                self.nodes[parent_idx].grad += grad;
            };
        }
    }
}


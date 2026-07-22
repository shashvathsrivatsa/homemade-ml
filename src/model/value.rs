use crate::*;


// ——— Value ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Pool {
    pub nodes: Vec<Node>,
    pub param_end: usize,
    pub input_end: usize,
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
        Pool { nodes: Vec::new(), param_end: 0, input_end: 0 }
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

    // —— Operations ———————————————————————————————————————————————————————————————————————
    pub fn add(&mut self, a: Value, b: Value) -> Value {
        let data = self.get_data(a) + self.get_data(b);
        self.new_kid(data, vec![a.0, b.0], "+")
    }

    pub fn sub(&mut self, a: Value, b: Value) -> Value {
        let data = self.get_data(a) - self.get_data(b);
        self.new_kid(data, vec![a.0, b.0], "-")
    }

    pub fn mul(&mut self, a: Value, b: Value) -> Value {
        let data = self.get_data(a) * self.get_data(b);
        self.new_kid(data, vec![a.0, b.0], "*")
    }

    pub fn div(&mut self, a: Value, b: Value) -> Value {
        let data = self.get_data(a) / self.get_data(b);
        self.new_kid(data, vec![a.0, b.0], "/")
    }

    pub fn neg(&mut self, a: Value) -> Value {
        self.new_kid(-self.get_data(a), vec![a.0], "neg")
    }

    pub fn pow2(&mut self, a: Value) -> Value {
        self.new_kid(self.get_data(a).powi(2), vec![a.0], "pow2")
    }

    pub fn exp(&mut self, a: Value) -> Value {
        self.new_kid(self.get_data(a).exp(), vec![a.0], "exp")
    }

    pub fn log(&mut self, a: Value) -> Value {
        self.new_kid(self.get_data(a).ln(), vec![a.0], "log")
    }

    pub fn max(&mut self, a: Value, b: Value) -> Value {
        self.new_kid(f64::max(self.get_data(a), self.get_data(b)), vec![a.0, b.0], "max")
    }

    // —— Optimize —————————————————————————————————————————————————————————————————————————
    pub fn set_param_end(&mut self) {
        self.param_end = self.nodes.len();
        param_print(self.nodes.len());
    }

    pub fn set_input_end(&mut self) {
        self.input_end = self.nodes.len();
    }

    pub fn flush_compute(&mut self) {
        if self.input_end == 0 { panic!("input end not set yet"); }
        self.nodes.truncate(self.input_end);
    }

    pub fn flush(&mut self) {
        if self.param_end == 0 { panic!("param end not set yet"); }
        self.nodes.truncate(self.param_end);
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
                let cur_grad = self.nodes[i].grad;
                let cur_data = self.nodes[i].data;
                let par_data = self.nodes[self.nodes[i].parents[p]].data;
                let other_par_data = if self.nodes[i].parents.len() > 1 {
                    self.nodes[self.nodes[i].parents[(p+1) % 2]].data
                } else { f64::MAX };

                let grad = match self.nodes[i].op {
                    "+"    => 1.0,
                    "-"    => if p == 0 { 1.0 } else { -1.0 },
                    "*"    => other_par_data,
                    "/"    => if p == 0 { 1.0 / other_par_data } else { -other_par_data / par_data.powi(2) },
                    "neg"  => -1.0,
                    "pow2" => 2.0 * par_data,
                    "exp"  => cur_data,
                    "log"  => 1.0 / par_data,
                    "max"  => if par_data > other_par_data { 1.0 } else { 0.0 },
                    op => { println!("{} not accounted for", op); 0.0 }
                } * cur_grad;

                let parent_idx = self.nodes[i].parents[p];
                self.nodes[parent_idx].grad += grad;
            };
        }
    }
}

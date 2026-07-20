use crate::*;


// ——— Value ——————————————————————————————————————————————————————————————————————————————————————————————————————————

#[derive(Clone)]
pub struct Value {
    pub id: usize,
    pub label: String,
    pub data: f64,
    pub grad: Rc<RefCell<f64>>,
    pub parents: Vec<Value>,
    pub op: &'static str,
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}


impl Add for &Value {
    type Output = Value;
    fn add(self, other: &Value) -> Value {
        Value::new_kid(self.data + other.data, vec![self.clone(), other.clone()], "+")
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        Value::new_kid(self.data + other.data, vec![self, other], "+")
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, other: &Value) -> Value {
        Value::new_kid(self.data * other.data, vec![self.clone(), other.clone()], "*")
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, other: &Value) -> Value {
        Value::new_kid(self.data - other.data, vec![self.clone(), other.clone()], "-")
    }
}

impl Value {

    // —— New ——————————————————————————————————————————————————————————————————————————————
    pub fn new(data: f64) -> Self {
        Self {
            id: next_id(),
            label: "".to_string(),
            data,
            grad: Rc::new(RefCell::new(0.0)),
            parents: Vec::new(),
            op: "",
        }
    }

    pub fn new_kid(data: f64, parents: Vec<Value>, op: &'static str) -> Self {
        Self {
            id: next_id(),
            label: "".to_string(),
            data,
            grad: Rc::new(RefCell::new(0.0)),
            parents,
            op,
        }
    }

    // —— Edit —————————————————————————————————————————————————————————————————————————————
    pub fn label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }

    pub fn set_grad(&mut self, grad: f64) {
        *self.grad.borrow_mut() = grad;
    }

    pub fn get_grad(&self) -> f64 {
        *self.grad.borrow()
    }

    // —— Backpropogate ————————————————————————————————————————————————————————————————————
    pub fn backpropogate(&mut self) {

        // Build topo order
        let mut order: Vec<Value> = Vec::new();
        let mut traversed: HashSet<usize> = HashSet::new();
        self.build_topo(&mut order, &mut traversed);

        // Initial node
        self.set_grad(1.0);

        // Run backwards through each node and feed both of its parents
        order.iter_mut().rev().for_each(|node| {
            for i in 0..node.parents.len() {
                let grad = match node.op {
                    "+"    => node.get_grad(),
                    "*"    => node.parents[(i+1) % 2].data * node.get_grad(),
                    "-"    => if i == 0 { node.get_grad() } else { -node.get_grad() },
                    "pow2" => 2.0 * node.parents[0].data * node.get_grad(),
                    "tanh" => (1.0 - node.data.powi(2)) * node.get_grad(),
                    op => { println!("{} not accounted for", op); 0.0 }
                };

                let parent = &mut node.parents[i];
                parent.set_grad(parent.get_grad() + grad);
            };
        });
    }

    fn build_topo(&mut self, order: &mut Vec<Value>, traversed: &mut HashSet<usize>) {
        if traversed.contains(&self.id) { return; }
        traversed.insert(self.id);

        // Zero grads
        self.set_grad(0.0);

        // Post order
        self.parents.iter_mut().for_each(|parent| parent.build_topo(order, traversed));
        order.push(self.clone());
    }

    // —— Other ops ————————————————————————————————————————————————————————————————————————
    pub fn tanh(&self) -> Value {
        let x = self.data;
        let t = ( (2.0 * x).exp() - 1.0 ) / ( (2.0 * x).exp() + 1.0 );
        Value::new_kid(t, vec![self.clone()], "tanh")
    }

    pub fn pow2(&mut self) -> Value {
        Value::new_kid(self.data.powi(2), vec![self.clone()], "pow2")
    }

}


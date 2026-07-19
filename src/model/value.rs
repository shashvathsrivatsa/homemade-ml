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
    // Base case: self's grad != 0
    // NOTE: this does NOT work for MLP
    pub fn backpropogate(&mut self, visited: &mut HashSet<usize>) {
        if visited.contains(&self.id) { return; }
        visited.insert(self.id);

        for i in 0..self.parents.len() {
            let grad = match self.op {
                "+"    => self.get_grad(),
                "*"    => self.parents[(i+1) % 2].data * self.get_grad(),
                "tanh" => (1.0 - self.data.powi(2)) * self.get_grad(),
                _ => 0.0
            };

            let p = &mut self.parents[i];
            p.set_grad(p.get_grad() + grad);
            p.backpropogate(visited);
        };
    }

    // —— Other ops ————————————————————————————————————————————————————————————————————————
    pub fn tanh(&self) -> Value {
        let x = self.data;
        let t = ( (2.0 * x).exp() - 1.0 ) / ( (2.0 * x).exp() + 1.0 );
        Value::new_kid(t, vec![self.clone()], "tanh")
    }

    pub fn powi(&mut self, p: i32) -> Value {
        Value::new_kid(self.data.powi(p), vec![self.clone()], "pow")
    }

}


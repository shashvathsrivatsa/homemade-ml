use crate::*;


// ——— Value Ops ——————————————————————————————————————————————————————————————————————————————————————————————————————

impl Pool {
    pub fn add(&mut self, a: Value, b: Value) -> Value {
        let data = self.get_data(a) + self.get_data(b);
        self.new_kid(data, vec![a.0, b.0], "+")
    }

    pub fn mul(&mut self, a: Value, b: Value) -> Value {
        let data = self.get_data(a) * self.get_data(b);
        self.new_kid(data, vec![a.0, b.0], "*")
    }

    pub fn sub(&mut self, a: Value, b: Value) -> Value {
        let data = self.get_data(a) - self.get_data(b);
        self.new_kid(data, vec![a.0, b.0], "-")
    }

    pub fn tanh(&mut self, a: Value) -> Value {
        let x = self.get_data(a);
        let t = ( (2.0 * x).exp() - 1.0 ) / ( (2.0 * x).exp() + 1.0 );
        self.new_kid(t, vec![a.0], "tanh")
    }

    pub fn pow2(&mut self, a: Value) -> Value {
        self.new_kid(self.get_data(a).powi(2), vec![a.0], "pow2")
    }
}

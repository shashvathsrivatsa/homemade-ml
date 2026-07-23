// use crate::*;


// ——— Tensor —————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct TensorNode {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub grad: Vec<f64>,
    pub parents: Vec<usize>,
    pub op: &'static str,
}

impl TensorNode {

    // —— Constructors —————————————————————————————————————————————————————————————————————
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        let grad: Vec<f64> = vec![0.0; data.len()];
        TensorNode { data, shape, strides, grad, parents: vec![], op: "" }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let data: Vec<f64> = (0..shape.iter().product()).map(|_| 0.0).collect();
        let strides = Self::compute_strides(&shape);
        let grad: Vec<f64> = vec![0.0; data.len()];
        TensorNode { data, shape, strides, grad, parents: vec![], op: "" }
    }

    // —— Helpers ——————————————————————————————————————————————————————————————————————————
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        (0..shape.len()).map(|dim| {
            (dim+1..shape.len()).fold(1, |acc, i| acc * shape[i])
        }).collect()
    }

    fn flat_index(&self, indices: &[usize]) -> usize {
        indices.iter().enumerate()
            .map(|(dim, &index)| index * self.strides[dim])
            .sum()
    }

    // —— Get / set ————————————————————————————————————————————————————————————————————————
    pub fn get(&self, indices: &[usize]) -> f64 {
        self.data[self.flat_index(indices)]
    }

    pub fn set(&mut self, indices: &[usize], new_value: f64) {
        let flat_index = self.flat_index(indices);
        self.data[flat_index] = new_value;
    }
}

// —— Real forward ops —————————————————————————————————————————————————————————————————————

pub fn transpose(a: &TensorNode) -> TensorNode {
    let data: Vec<f64> = (0..a.shape[1]).flat_map(|col| {
        (0..a.shape[0]).map(move |row| a.get(&[row, col]))
    }).collect();
    TensorNode::new(data, vec![a.shape[1], a.shape[0]])
}

pub fn matmul(a: &TensorNode, b: &TensorNode) -> TensorNode {
    assert_eq!(a.shape[1], b.shape[0], "invalid dimensions for matmul");
    let mut c: Vec<f64> = vec![];

    for row in 0..a.shape[0] {
        for col in 0..b.shape[1] {
            let c_n = (0..a.shape[1]).fold(0.0, |acc, n| {
                acc + a.get(&[row, n]) * b.get(&[n, col])
            });
            c.push(c_n);
        }
    }

    TensorNode::new(c, vec![a.shape[0], b.shape[1]])
}

pub fn matadd(a: &TensorNode, b: &TensorNode) -> TensorNode {
    assert_eq!(a.shape, b.shape, "invalid dimensions for matadd");

    let c: Vec<f64> = a.data.iter().zip(b.data.iter())
        .map(|(a_k, b_k)| a_k + b_k)
        .collect();

    TensorNode::new(c, a.shape.clone())
}


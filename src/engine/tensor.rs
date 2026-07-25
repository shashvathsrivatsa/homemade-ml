use crate::*;


// ——— Tensor —————————————————————————————————————————————————————————————————————————————————————————————————————————

use rand::{Rng, thread_rng};

pub struct TensorNode {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub grad: Vec<f32>,
    pub parents: Vec<usize>,
    pub op: &'static str,
}

impl TensorNode {

    // —— Constructors —————————————————————————————————————————————————————————————————————
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        let grad: Vec<f32> = vec![0.0; data.len()];
        TensorNode { data, shape, strides, grad, parents: vec![], op: "" }
    }

    pub fn fill(shape: Vec<usize>, num: f32) -> Self {
        let data: Vec<f32> = vec![num; shape.iter().product()];
        let strides = Self::compute_strides(&shape);
        let grad: Vec<f32> = vec![0.0; data.len()];
        TensorNode { data, shape, strides, grad, parents: vec![], op: "" }
    }

    pub fn new_rand(shape: Vec<usize>) -> Self {
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..shape.iter().product()).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let strides = Self::compute_strides(&shape);
        let grad: Vec<f32> = vec![0.0; data.len()];
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
    pub fn get(&self, indices: &[usize]) -> f32 {
        self.data[self.flat_index(indices)]
    }

    pub fn set(&mut self, indices: &[usize], v: f32) {
        let flat_index = self.flat_index(indices);
        self.data[flat_index] = v;
    }

    pub fn set_grad(&mut self, grad_value: f32) {
        self.grad = self.grad.iter().map(|_| grad_value).collect();
    }
}

// —— Real forward ops —————————————————————————————————————————————————————————————————————

pub fn transpose(a: &TensorNode) -> TensorNode {
    let data: Vec<f32> = (0..a.shape[1]).flat_map(|col| {
        (0..a.shape[0]).map(move |row| a.get(&[row, col]))
    }).collect();
    TensorNode::new(data, vec![a.shape[1], a.shape[0]])
}

pub fn matadd(a: &TensorNode, b: &TensorNode) -> TensorNode {
    assert_eq!(a.shape, b.shape, "invalid dimensions for matadd");

    let c: Vec<f32> = a.data.iter().zip(b.data.iter())
        .map(|(a_k, b_k)| a_k + b_k)
        .collect();

    TensorNode::new(c, a.shape.clone())
}


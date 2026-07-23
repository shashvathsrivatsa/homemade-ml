// use crate::*;


// ——— Tensor —————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {

    // —— Constructors —————————————————————————————————————————————————————————————————————
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Tensor { data, shape, strides }

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


pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
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

    Tensor::new(c, vec![a.shape[0], b.shape[1]])
}


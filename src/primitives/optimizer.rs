use crate::*;

// ——— Optimizer —————————————————————————————————————————————————————————————————————————————————————————————————————

pub enum Optimizer {
    SGD,
    AdamOptimizer,
}

pub use Optimizer::*;


pub enum OptimizerData {
    SGD(SGD),
    Adam(Vec<AdamOptimizer>),
}

// —— SGD —————————————————————————————————————————————————————————————————————————————————
pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    pub fn step(&self, gpu: &Gpu, t: &mut TensorNode) {
        let out = gpu.empty_buffer(t.len);
        gpu.dispatch(
            "update",
            &[&t.data, &t.grad],
            &[&out],
            [t.len as u32, self.lr.to_bits(), 0, 0],
            Pool::groups_1d(t.len),
        );
        t.data = out;
    }
}

// —— Adam —————————————————————————————————————————————————————————————————————————————————
pub struct AdamOptimizer {
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub t: i32,
    pub m: wgpu::Buffer,
    pub v: wgpu::Buffer,
}

impl AdamOptimizer {
    pub fn new(lr: f32, gpu: &Gpu, len: usize) -> Self {
        Self {
            lr,
            b1: 0.9,
            b2: 0.999,
            t: 0,
            m: gpu.zero_buffer(len),
            v: gpu.zero_buffer(len),
        }
    }

    pub fn step(&mut self, gpu: &Gpu, t: &mut TensorNode) {
        self.t += 1;
        let out = gpu.empty_buffer(t.len);
        gpu.dispatch(
            "adam",
            &[&t.data, &t.grad],
            &[&self.m, &self.v, &out],
            [t.len as u32, self.lr.to_bits(), self.t as u32, 0],
            Pool::groups_1d(t.len),
        );
        t.data = out;
    }
}

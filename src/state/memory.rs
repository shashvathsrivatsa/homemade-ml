// use crate::*;

// ——— Memory —————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct MemoryNode {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

pub struct Memory {
    pub memories: Vec<MemoryNode>,
    pub capacity: usize,
}

impl Memory {
    pub fn new() -> Self {
        Self { memories: vec![], capacity: 10_000 }
    }

    pub fn push(&mut self, node: MemoryNode) {
        if self.memories.len() >= self.capacity {
            self.memories.remove(0);
        }
        self.memories.push(node);
    }
}


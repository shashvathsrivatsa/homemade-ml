use crate::*;

// ——— Memory —————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Experience {
    pub state: Vec<f32>,
    pub action: usize,
    pub next_state: Vec<f32>,
    pub reward: f32,
    pub done: bool,
}

pub struct Memory {
    pub memories: VecDeque<Experience>,
    pub capacity: usize,
}

impl Memory {
    pub fn new() -> Self {
        Self {
            memories: VecDeque::with_capacity(10_000),
            capacity: 10_000,
        }
    }

    pub fn push(&mut self, node: Experience) {
        if self.memories.len() >= self.capacity {
            self.memories.pop_front();
        }
        self.memories.push_back(node);
    }
}


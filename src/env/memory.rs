use crate::*;

// ——— Memory —————————————————————————————————————————————————————————————————————————————————————————————————————————

#[derive(Clone)]
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
    pub batch_size: usize,
}

impl Memory {
    pub fn new(memory_capacity: usize, batch_size: usize) -> Self {
        Self {
            memories: VecDeque::with_capacity(memory_capacity),
            capacity: memory_capacity,
            batch_size,
        }
    }

    pub fn push(&mut self, node: Experience) {
        if self.memories.len() >= self.capacity {
            self.memories.pop_front();
        }
        self.memories.push_back(node);
    }

    pub fn batch(&self) -> Vec<&Experience> {
        let mut rng = thread_rng();
        (0..self.batch_size)
            .map(|_| &self.memories[rng.gen_range(0..self.memories.len())])
            .collect()
    }
}


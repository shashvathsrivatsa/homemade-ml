use crate::*;

pub mod loss_graph;
pub use loss_graph::*;

pub mod fruits_graph;
pub use fruits_graph::*;

pub mod cart_pole_visualizaiton;
pub use cart_pole_visualizaiton::*;

pub mod snake_visualization;
pub use snake_visualization::*;

pub trait StateVisualization {
    fn update_state(
        &mut self,
        segments: &VecDeque<(f32, f32)>,
        fruit: (f32, f32),
        fruits_eaten: usize,
        done: bool,
    ) -> std::io::Result<bool>;
}

pub mod loss_graph;
pub use loss_graph::*;

pub mod fruits_graph;
pub use fruits_graph::*;

pub mod cart_pole_visualizaiton;
pub use cart_pole_visualizaiton::*;

pub trait StateVisualization {
    fn update_state(&mut self, fruits_eaten: usize, done: bool) -> std::io::Result<bool>;
}

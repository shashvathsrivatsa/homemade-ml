pub mod loss_graph;
pub use loss_graph::*;

pub mod episode_graph;
pub use episode_graph::*;

pub mod cart_pole_visualizaiton;
pub use cart_pole_visualizaiton::*;

pub trait StateVisualization {
    fn update_state(
        &mut self,
        cart_x: f32,
        pole_angle: f32,
        episode_len: usize,
        done: bool,
    ) -> std::io::Result<bool>;
}

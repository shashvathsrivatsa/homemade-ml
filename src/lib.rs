pub mod utils;
pub use utils::*;

pub mod model;
pub use model::*;

pub mod draw_dot;
pub use draw_dot::*;

pub use std::fmt;
pub use std::ops::{ Add, Mul, Sub };
pub use std::rc::Rc;
pub use std::cell::RefCell;
pub use std::collections::{ HashMap, HashSet, VecDeque };
pub use std::sync::atomic::AtomicUsize;

pub use plotters::prelude::*;
use rand::Rng;


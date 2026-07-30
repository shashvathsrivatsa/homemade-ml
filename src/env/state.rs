use crate::*;

// ——— State ——————————————————————————————————————————————————————————————————————————————————————————————————————————

const G: f32 = 9.8;
const M_C: f32 = 1.0;
const M_P: f32 = 0.1;
const L_P: f32 = 1.0;
const F_FACTOR: f32 = 10.0;

pub struct State {
    pub cart_x: f32,
    pub cart_v: f32,
    pub pole_angle: f32,
    pub pole_angular_v: f32,
    pub episode_length: usize,
}

impl State {
    pub fn new() -> Self {
        Self {
            cart_x: 0.0,
            cart_v: 0.0,
            pole_angle: 0.05,
            pole_angular_v: 0.0,
            episode_length: 0,
        }
    }

    pub fn step(&mut self, action: usize, memory: &mut Memory) {
        self.episode_length += 1;

        // Init
        let dir: i32 = (action as i32 + 1) * 2 - 3;
        let f_push = dir as f32 * F_FACTOR;
        let theta = self.pole_angle;
        let omega = self.pole_angular_v;
        let dt = 0.02;

        // Compute
        let temp = (f_push + M_P * L_P / 2.0 * omega.powi(2) * theta.sin()) / (M_C + M_P);
        let temp2 = L_P / 2.0 * (4.0 / 3.0 - M_P * theta.cos().powi(2) / (M_C + M_P));
        let alpha = (G * theta.sin() - theta.cos() * temp) / temp2;
        let a_c = temp - M_P * L_P / 2.0 * alpha * theta.cos() / (M_C + M_P);

        // Finals
        let pole_angular_v_f = self.pole_angular_v + alpha * dt;
        let pole_angle_f = self.pole_angle + self.pole_angular_v * dt;
        let cart_v_f = self.cart_v + a_c * dt;
        let cart_x_f = self.cart_x + self.cart_v * dt;
        let done = self.cart_x.abs() > 5.0 || self.pole_angle.abs() > 12.0_f32.to_radians();

        // Update
        let next_state = vec![cart_x_f, cart_v_f, pole_angle_f, pole_angular_v_f];
        memory.push(Experience { state: self.to_vec(), action, next_state: next_state.clone(), reward: 1.0, done });
        let old_state = [&mut self.cart_x, &mut self.cart_v, &mut self.pole_angle, &mut self.pole_angular_v];
        old_state.into_iter().zip(next_state.iter()).for_each(|(v_i, &v_f)| *v_i = v_f);

        // Reset if done
        if done {
            println!("{}", self.episode_length);
            *self = State::new();
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.cart_x, self.cart_v, self.pole_angle, self.pole_angular_v]
    }
}


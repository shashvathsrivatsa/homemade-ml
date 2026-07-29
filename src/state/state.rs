// use crate::*;

// ——— State ——————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct State {
    pub m_c: f32,
    pub m_p: f32,
    pub l_p: f32,

    pub cart_x: f32,
    pub cart_v: f32,
    pub pole_angle: f32,
    pub pole_angular_v: f32,
}

impl State {
    pub fn new() -> Self {
        Self {
            m_c: 1.0,
            m_p: 0.1,
            l_p: 1.0,

            cart_x: 0.0,
            cart_v: 0.0,
            pole_angle: 90.0,
            pole_angular_v: 0.0,
        }
    }

    pub fn step(&mut self, f_push: f32) -> bool {
        let g = 9.8;
        let l = self.l_p / 2.0;
        let m = self.m_p;
        let mc = self.m_c;
        let theta = self.pole_angle;
        let omega = self.pole_angular_v;
        let dt = 0.02;

        let temp = (f_push + m * l * omega.powi(2) * theta.sin()) / (mc + m);
        let temp2 = l * (4.0 / 3.0 - m * theta.cos().powi(2) / (mc + m));
        let alpha = (g * theta.sin() - theta.cos() * temp) / temp2;
        let a_c = temp - m * l * alpha * theta.cos() / (mc + m);

        self.pole_angular_v += alpha * dt;
        self.pole_angle += self.pole_angular_v * dt;
        self.cart_v += a_c * dt;
        self.cart_x += self.cart_v * dt;

        self.cart_x.abs() > 5.0 || self.pole_angle.abs() > 12.0_f32.to_radians()
    }
}


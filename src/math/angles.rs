pub fn clamp_deg(deg: f32) -> f32 {
    (deg % 360.0 + 360.0) % 360.0
}

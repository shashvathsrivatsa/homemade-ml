use crate::*;

pub fn polar_to_cart(r: f32, theta: f32) -> (f32, f32) {
    let theta = theta.to_radians();
    (r * f32::cos(theta), r * f32::sin(theta))
}

pub fn polar_to_point(
    p1: (f32, f32),
    heading: f32,
    p2: (f32, f32),
) -> (f32, f32) {

    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let r = (dx.powi(2) + dy.powi(2)).sqrt();
    let theta = dy.atan2(dx).to_degrees();
    let rel_theta = clamp_deg(theta - heading);
    (r, rel_theta)
}

pub fn polar_to_seg(
    p1: (f32, f32),
    heading: f32,
    seg: ((f32, f32), (f32, f32)),
) -> (f32, f32) {

    let (a, b) = seg;
    let ab = (b.0 - a.0, b.1 - a.1);
    let ap = (p1.0 - a.0, p1.1 - a.1);
    let t = ((ap.0 * ab.0 + ap.1 * ab.1) / (ab.0 * ab.0 + ab.1 * ab.1)).clamp(0.0, 1.0);
    let closest = (a.0 + t * ab.0, a.1 + t * ab.1);
    polar_to_point(p1, heading, closest)
}

pub fn raycast(
    origin: (f32, f32),
    heading: f32,
    max_dist: f32,
    segments: impl Iterator<Item = ((f32, f32), (f32, f32))>,
) -> f32 {
    let (dx, dy) = polar_to_cart(max_dist, heading);
    let ray = (origin, (origin.0 + dx, origin.1 + dy));
    
    segments
        .filter_map(|seg| seg_intersect_dist(ray, seg))
        .fold(f32::MAX, f32::min)
        .min(max_dist)
}


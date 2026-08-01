pub fn seg_intersect(seg_1: ((f32, f32), (f32, f32)), seg_2: ((f32, f32), (f32, f32))) -> bool {
    let (a, b) = seg_1;
    let (c, d) = seg_2;
    
    let cross = |o: (f32,f32), a: (f32,f32), b: (f32,f32)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    
    let d1 = cross(c, d, a);
    let d2 = cross(c, d, b);
    let d3 = cross(a, b, c);
    let d4 = cross(a, b, d);
    
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) &&
       ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)) {
        return true;
    }
    
    false
}

pub fn cir_intersect(point: (f32, f32), cir_coords: (f32, f32), r: f32) -> bool {
    let dist_from_cir = (point.0 - cir_coords.0).powi(2) + (point.1 - cir_coords.1).powi(2);
    dist_from_cir < r.powi(2)
}


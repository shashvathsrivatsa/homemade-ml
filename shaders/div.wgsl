struct Params { rows: u32, cols: u32, unused_0: u32, unused_1: u32 };

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn div(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i < p.rows * p.cols {
        out[i] = a[i] / b[i / p.cols];
    }
}

@compute @workgroup_size(256)
fn div_backward_a(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i < p.rows * p.cols {
        out[i] = b[i] / a[i / p.cols];
    }
}

// `out` initially contains the upstream gradient.
@compute @workgroup_size(256)
fn div_backward_b(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if row >= p.rows { return; }
    var value = 0.0;
    for (var col = 0u; col < p.cols; col++) {
        let i = row * p.cols + col;
        value += a[i] * out[i];
    }
    out[row] = -value / (b[row] * b[row]);
}

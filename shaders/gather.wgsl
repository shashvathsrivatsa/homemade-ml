struct Params { rows: u32, cols: u32, unused_0: u32, unused_1: u32 };

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.y * 65535u * 256u + id.x;
    if row >= p.rows { return; }
    out[row] = input[row * p.cols + u32(labels[row])];
}

@compute @workgroup_size(256)
fn gather_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.y * 65535u * 256u + id.x;
    if i >= p.rows * p.cols { return; }
    let row = i / p.cols;
    out[i] = select(0.0, input[row], i % p.cols == u32(labels[row]));
}

struct Params { rows: u32, cols: u32, unused: u32, sentinel: u32 };

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn bias_add(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.y * 65535u * 256u + id.x;
    if i >= p.rows * p.cols { return; }
    out[i] = input[i] + bias[i % p.cols];
}

@compute @workgroup_size(256)
fn bias_add_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.y * 65535u * 256u + id.x;
    if col >= p.cols { return; }
    var value = 0.0;
    for (var row = 0u; row < p.rows; row++) {
        value += input[row * p.cols + col];
    }
    out[col] = value;
    if p.sentinel == 4294967295u { out[col] = bias[0]; }
}

struct Params { len: u32, learning_rate_bits: u32, unused_0: u32, unused_1: u32 };

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.y * 65535u * 256u + id.x;
    if i >= p.len { return; }
    out[i] = data[i] - bitcast<f32>(p.learning_rate_bits) * grad[i];
}

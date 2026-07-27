struct Params { len: u32, op: u32, unused_0: u32, unused_1: u32 };

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// op: 0 add, 1 mul, 2 max, 3 mul_backward_a, 4 mul_backward_b,
// 5 max_backward_a, 6 max_backward_b
@compute @workgroup_size(256)
fn binary(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.y * 65535u * 256u + id.x;
    if i >= p.len { return; }

    switch p.op {
        case 0u: { out[i] = a[i] + b[i]; }
        case 1u: { out[i] = a[i] * b[i]; }
        case 2u: { out[i] = max(a[i], b[i]); }
        case 3u: { out[i] = b[i] * out[i]; }
        case 4u: { out[i] = a[i] * out[i]; }
        case 5u: { out[i] = select(0.0, out[i], a[i] > b[i]); }
        default: { out[i] = select(out[i], 0.0, a[i] > b[i]); }
    }
}

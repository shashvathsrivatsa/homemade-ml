struct Params { len: u32, lr_bits: u32, t: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(256)
fn adam(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.y * 65535u * 256u + id.x;
    if idx >= p.len { return; }

    let b1 = 0.9;
    let b2 = 0.999;
    let lr = bitcast<f32>(p.lr_bits);
    let t = f32(p.t);

    let g = grads[idx];
    let mi = b1 * m[idx] + (1.0 - b1) * g;
    let vi = b2 * v[idx] + (1.0 - b2) * g * g;
    m[idx] = mi;
    v[idx] = vi;

    let m_hat = mi / (1.0 - pow(b1, t));
    let v_hat = vi / (1.0 - pow(b2, t));
    out[idx] = params[idx] - lr * m_hat / (sqrt(v_hat) + 1e-8);
}

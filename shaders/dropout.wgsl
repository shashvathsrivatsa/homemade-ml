struct Params { len: u32, rate_bits: u32, seed: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

fn rand(seed: u32) -> f32 {
    var s = seed ^ 2747636419u;
    s = s * 2654435769u;
    s ^= s >> 16u;
    s = s * 2654435769u;
    s ^= s >> 16u;
    s = s * 2654435769u;
    return f32(s) / 4294967295.0;
}

@compute @workgroup_size(256)
fn dropout(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * 65535u * 256u + gid.x;
    if idx >= p.len { return; }
    let rate = bitcast<f32>(p.rate_bits);
    let keep = select(0.0, 1.0, rand(idx ^ p.seed) >= rate);
    let scaled_mask = keep / (1.0 - rate);
    mask[idx] = scaled_mask;
    out[idx] = inp[idx] * scaled_mask;
}

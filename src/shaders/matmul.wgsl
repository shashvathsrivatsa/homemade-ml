struct Params { m: u32, k: u32, n: u32, transpose: u32 };

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    if row >= p.m || col >= p.n { return; }

    var value = 0.0;
    for (var inner = 0u; inner < p.k; inner++) {
        let a_index = select(row * p.k + inner, inner * p.m + row, (p.transpose & 1u) != 0u);
        let b_index = select(inner * p.n + col, col * p.k + inner, (p.transpose & 2u) != 0u);
        value += a[a_index] * b[b_index];
    }
    out[row * p.n + col] = value;
}

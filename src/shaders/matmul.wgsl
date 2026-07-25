// matmul.wgsl
struct Dims {
    M: u32,  // rows of A
    K: u32,  // cols of A / rows of B
    N: u32,  // cols of B
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    if row >= dims.M || col >= dims.N { return; }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.K; k++) {
        sum += A[row * dims.K + k] * B[k * dims.N + col];
    }
    C[row * dims.N + col] = sum;
}

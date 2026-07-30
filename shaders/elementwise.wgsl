struct Params { len: u32, op: u32, unused_0: u32, unused_1: u32 };

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// op: 0 tanh, 1 exp, 2 log, 3 neg, 4 relu, 5 exp_backward,
// 6 log_backward, 7 neg_backward, 8 tanh_backward, 9 copy, 10 scalar broadcast,
// 11 dropout_backward, 12 sq, 13 sq_backward
@compute @workgroup_size(256)
fn unary(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.y * 65535u * 256u + id.x;
    if i >= p.len { return; }

    switch p.op {
        case 0u: { out[i] = tanh(input[i]); }
        case 1u: { out[i] = exp(input[i]); }
        case 2u: { out[i] = log(input[i]); }
        case 3u: { out[i] = -input[i]; }
        case 4u: { out[i] = max(input[i], 0.0); }
        case 5u: { out[i] = input[i] * grad[i]; }
        case 6u: { out[i] = grad[i] / input[i]; }
        case 7u: { out[i] = -input[i]; }
        case 8u: { out[i] = (1.0 - input[i] * input[i]) * grad[i]; }
        case 10u: { out[i] = input[0]; }
        case 11u: { out[i] = select(0.0, grad[i], input[i] != 0.0); }
        case 12u: { out[i] = input[i] * input[i]; }
        case 13u: { out[i] = 2.0 * input[i] * grad[i]; }
        default: { out[i] = input[i]; }
    }
}

struct Params { rows: u32, cols: u32, unused: u32, sentinel: u32 };

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn sum(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if row >= p.rows { return; }
    var value = 0.0;
    for (var col = 0u; col < p.cols; col++) {
        value += input[row * p.cols + col];
    }
    out[row] = value;
    if p.sentinel == 4294967295u { out[row] = grad[0]; }
}

@compute @workgroup_size(256)
fn sub_row_max(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if row >= p.rows { return; }
    var row_max = -3.402823e38;
    for (var col = 0u; col < p.cols; col++) {
        row_max = max(row_max, input[row * p.cols + col]);
    }
    for (var col = 0u; col < p.cols; col++) {
        out[row * p.cols + col] = input[row * p.cols + col] - row_max;
    }
    if p.sentinel == 4294967295u { out[row * p.cols] = grad[0]; }
}

@compute @workgroup_size(256)
fn mean_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i < p.rows {
        out[i] = input[0] / f32(p.rows);
        if p.sentinel == 4294967295u { out[i] = grad[0]; }
    }
}

@compute @workgroup_size(256)
fn sum_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i < p.rows * p.cols {
        out[i] = input[i / p.cols];
        if p.sentinel == 4294967295u { out[i] = grad[0]; }
    }
}

@compute @workgroup_size(256)
fn sub_row_max_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if row >= p.rows { return; }

    var row_max = -3.402823e38;
    var argmax = 0u;
    var grad_sum = 0.0;
    for (var col = 0u; col < p.cols; col++) {
        let i = row * p.cols + col;
        grad_sum += grad[i];
        if input[i] > row_max {
            row_max = input[i];
            argmax = col;
        }
    }
    for (var col = 0u; col < p.cols; col++) {
        let i = row * p.cols + col;
        out[i] = grad[i] - select(0.0, grad_sum, col == argmax);
    }
}

struct Params { a: u32, b: u32, c: u32, d: u32 };

@group(0) @binding(0) var<storage, read> in0: array<f32>;
@group(0) @binding(1) var<storage, read> in1: array<f32>;
@group(0) @binding(2) var<storage, read_write> out0: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x; let col = id.y;
    if row >= p.a || col >= p.c { return; }
    var v = 0.0;
    for (var k = 0u; k < p.b; k++) {
        let ai = select(row * p.b + k, k * p.a + row, (p.d & 1u) != 0u);
        let bi = select(k * p.c + col, col * p.b + k, (p.d & 2u) != 0u);
        v += in0[ai] * in1[bi];
    }
    out0[row * p.c + col] = v;
}

// p.b: 0 tanh, 1 exp, 2 log, 3 neg, 4 relu, 5 exp_backward,
// 6 log_backward, 7 neg_backward, 8 tanh_backward, 9 copy, 10 scalar broadcast
@compute @workgroup_size(256)
fn unary(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; if i >= p.a { return; }
    switch p.b {
        case 0u: { out0[i] = tanh(in0[i]); }
        case 1u: { out0[i] = exp(in0[i]); }
        case 2u: { out0[i] = log(in0[i]); }
        case 3u: { out0[i] = -in0[i]; }
        case 4u: { out0[i] = max(in0[i], 0.0); }
        case 5u: { out0[i] = in0[i] * in1[i]; }
        case 6u: { out0[i] = in1[i] / in0[i]; }
        case 7u: { out0[i] = -in0[i]; }
        case 8u: { out0[i] = (1.0 - in0[i] * in0[i]) * in1[i]; }
        case 10u: { out0[i] = in0[0]; }
        default: { out0[i] = in0[i]; }
    }
}

// p.b: 0 add, 1 mul, 2 max, 3 mul_backward_a, 4 mul_backward_b,
// 5 max_backward_a, 6 max_backward_b
@compute @workgroup_size(256)
fn binary(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; if i >= p.a { return; }
    switch p.b {
        case 0u: { out0[i] = in0[i] + in1[i]; }
        case 1u: { out0[i] = in0[i] * in1[i]; }
        case 2u: { out0[i] = max(in0[i], in1[i]); }
        case 3u: { out0[i] = in1[i] * out0[i]; }
        case 4u: { out0[i] = in0[i] * out0[i]; }
        case 5u: { out0[i] = select(0.0, out0[i], in0[i] > in1[i]); }
        default: { out0[i] = select(out0[i], 0.0, in0[i] > in1[i]); }
    }
}

@compute @workgroup_size(256)
fn bias_add(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if i<p.a*p.b { out0[i]=in0[i]+in1[i%p.b]; }
}
@compute @workgroup_size(256)
fn div(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if i<p.a*p.b { out0[i]=in0[i]/in1[i/p.b]; }
}
@compute @workgroup_size(256)
fn sum(@builtin(global_invocation_id) id: vec3<u32>) {
    let r=id.x; if r>=p.a{return;} var v=0.0;
    for(var c=0u;c<p.b;c++){v+=in0[r*p.b+c];} out0[r]=v;
    if p.d==4294967295u { out0[r]=in1[0]; }
}
@compute @workgroup_size(256)
fn sub_row_max(@builtin(global_invocation_id) id: vec3<u32>) {
    let r=id.x; if r>=p.a{return;} var m=-3.402823e38;
    for(var c=0u;c<p.b;c++){m=max(m,in0[r*p.b+c]);}
    for(var c=0u;c<p.b;c++){out0[r*p.b+c]=in0[r*p.b+c]-m;}
    if p.d==4294967295u { out0[r*p.b]=in1[0]; }
}
@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let r=id.x; if r<p.a { out0[r]=in0[r*p.b+u32(in1[r])]; }
}
@compute @workgroup_size(256)
fn bias_add_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let c=id.x; if c>=p.b{return;} var v=0.0;
    for(var r=0u;r<p.a;r++){v+=in0[r*p.b+c];} out0[c]=v;
    if p.d==4294967295u { out0[c]=in1[0]; }
}
@compute @workgroup_size(256)
fn gather_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if i<p.a*p.b { let r=i/p.b; out0[i]=select(0.0,in0[r],i%p.b==u32(in1[r])); }
}
@compute @workgroup_size(256)
fn mean_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if i<p.a { out0[i]=in0[0]/f32(p.a); if p.d==4294967295u {out0[i]=in1[0];} }
}
@compute @workgroup_size(256)
fn sum_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if i<p.a*p.b { out0[i]=in0[i/p.b]; if p.d==4294967295u {out0[i]=in1[0];} }
}
@compute @workgroup_size(256)
fn div_backward_a(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if i<p.a*p.b { out0[i]=in1[i]/in0[i/p.b]; }
}
// in0=a, in1=b, out0 initially dc; one invocation per row.
@compute @workgroup_size(256)
fn div_backward_b(@builtin(global_invocation_id) id: vec3<u32>) {
    let r=id.x; if r>=p.a{return;} var v=0.0;
    for(var c=0u;c<p.b;c++){let i=r*p.b+c;v+=in0[i]*out0[i];}
    out0[r]=-v/(in1[r]*in1[r]);
}
// in0=input, in1=dc
@compute @workgroup_size(256)
fn sub_row_max_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let r=id.x; if r>=p.a{return;} var m=-3.402823e38; var arg=0u; var s=0.0;
    for(var c=0u;c<p.b;c++){let i=r*p.b+c;s+=in1[i];if(in0[i]>m){m=in0[i];arg=c;}}
    for(var c=0u;c<p.b;c++){let i=r*p.b+c;out0[i]=in1[i]-select(0.0,s,c==arg);}
}
@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if i<p.a { out0[i]=in0[i]-bitcast<f32>(p.b)*in1[i]; }
}

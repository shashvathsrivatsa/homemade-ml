use crate::*;


// ——— MatMul Gpu —————————————————————————————————————————————————————————————————————————————————————————————————————

pub fn matmul_gpu(gpu: &Gpu, a: &TensorNode, b: &TensorNode) -> TensorNode {

    // Extract data
    let m = a.shape[0] as u32;
    let k = a.shape[1] as u32;
    let n = b.shape[1] as u32;
    let a = &a.data;
    let b = &b.data;

    let t0 = Instant::now();

    // Upload A
    let a_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Upload B
    let b_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Output C
    let c_size = (m * n) as u64 * 4;
    let c_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: c_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Dims uniform
    let dims: [u32; 3] = [m, k, n];
    let dims_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Staging buffer to read results
    let staging_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: c_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Bind group
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &gpu.matmul_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
        ],
    });

    let t1 = Instant::now();

    // Encode and dispatch
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&gpu.matmul_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((m + 15) / 16, (n + 15) / 16, 1);
    }
    encoder.copy_buffer_to_buffer(&c_buf, 0, &staging_buf, 0, c_size);
    gpu.queue.submit(Some(encoder.finish()));

    let t2 = Instant::now();

    // Read back
    let slice = staging_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buf.unmap();

    let t3 = Instant::now();
    // println!("upload: {:?}, dispatch: {:?}, readback: {:?}", t1-t0, t2-t1, t3-t2);

    TensorNode::new(result, vec![m as usize, n as usize])
}

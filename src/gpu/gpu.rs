use crate::*;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
    pub one: wgpu::Buffer,
}

impl Gpu {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(target_os = "linux")]
            backends: wgpu::Backends::VULKAN,
            #[cfg(not(target_os = "linux"))]
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("no Vulkan GPU adapter found");
        println!("{:?}", adapter.get_info());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("failed to create wgpu device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor ops"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ops.wgsl").into()),
        });
        let entries = [
            "matmul",
            "unary",
            "binary",
            "bias_add",
            "div",
            "sum",
            "sub_row_max",
            "gather",
            "bias_add_backward",
            "gather_backward",
            "mean_backward",
            "div_backward_a",
            "div_backward_b",
            "sum_backward",
            "sub_row_max_backward",
            "update",
        ];
        let pipelines = entries
            .into_iter()
            .map(|entry| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &shader,
                    entry_point: entry,
                    compilation_options: Default::default(),
                    cache: None,
                });
                (entry, pipeline)
            })
            .collect();
        let one = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("constant one"),
            contents: bytemuck::bytes_of(&1.0_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Self {
            device,
            queue,
            pipelines,
            one,
        }
    }

    pub fn upload_buffer(&self, data: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    pub fn empty_buffer(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor"),
            size: (len.max(1) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn download_buffer(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let size = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("GPU readback failed");
        let mapped = slice.get_mapped_range();
        let result = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        result
    }

    pub fn copy_buffer(&self, source: &wgpu::Buffer, destination: &wgpu::Buffer, len: usize) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, destination, 0, (len * 4) as u64);
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn clear_buffer(&self, buffer: &wgpu::Buffer) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(buffer, 0, None);
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn dispatch(
        &self,
        entry: &'static str,
        inputs: &[&wgpu::Buffer],
        outputs: &[&wgpu::Buffer],
        params: [u32; 4],
        workgroups: (u32, u32, u32),
    ) {
        let pipeline = &self.pipelines[entry];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let entries: Vec<_> = inputs
            .iter()
            .chain(outputs.iter())
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .chain(std::iter::once(wgpu::BindGroupEntry {
                binding: (inputs.len() + outputs.len()) as u32,
                resource: params_buf.as_entire_binding(),
            }))
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

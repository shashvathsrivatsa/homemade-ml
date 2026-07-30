use crate::*;
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

struct ThreadWaker(std::thread::Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut context = Context::from_waker(&waker);

    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => std::thread::park(),
        }
    }
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
    pub one: wgpu::Buffer,
}

impl Gpu {
    pub fn new() -> Self {
        block_on(Self::new_async())
    }

    async fn new_async() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(target_os = "linux")]
            backends: wgpu::Backends::VULKAN,
            // #[cfg(not(target_os = "linux"))]
            // backends: wgpu::Backends::PRIMARY,
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
        // println!("{:?}", adapter.get_info());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 1024 * 1024 * 1024,  // 1GB
                    max_buffer_size: 1024 * 1024 * 1024,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            }, None)
        .await
            .expect("failed to create wgpu device");
        let shader_sources = [
            ("matmul", include_str!("../../shaders/matmul.wgsl")),
            ("elementwise", include_str!("../../shaders/elementwise.wgsl")),
            ("binary", include_str!("../../shaders/binary.wgsl")),
            ("reduce", include_str!("../../shaders/reduce.wgsl")),
            ("gather", include_str!("../../shaders/gather.wgsl")),
            ("bias", include_str!("../../shaders/bias.wgsl")),
            ("div", include_str!("../../shaders/div.wgsl")),
            ("optim", include_str!("../../shaders/optim.wgsl")),
            ("adam", include_str!("../../shaders/adam.wgsl")),
            ("dropout", include_str!("../../shaders/dropout.wgsl")),
        ];
        let shaders: std::collections::HashMap<_, _> = shader_sources
            .into_iter()
            .map(|(name, source)| {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
                (name, shader)
            })
        .collect();
        let pipeline_entries = [
            ("matmul", "matmul", "matmul"),
            ("unary", "elementwise", "unary"),
            ("binary", "binary", "binary"),
            ("sum", "reduce", "sum"),
            ("sub_row_max", "reduce", "sub_row_max"),
            ("mean_backward", "reduce", "mean_backward"),
            ("sum_backward", "reduce", "sum_backward"),
            ("sub_row_max_backward", "reduce", "sub_row_max_backward"),
            ("row_mean", "reduce", "row_mean"),
            ("row_mean_backward", "reduce", "row_mean_backward"),
            ("gather", "gather", "gather"),
            ("gather_backward", "gather", "gather_backward"),
            ("bias_add", "bias", "bias_add"),
            ("bias_add_backward", "bias", "bias_add_backward"),
            ("div", "div", "div"),
            ("div_backward_a", "div", "div_backward_a"),
            ("div_backward_b", "div", "div_backward_b"),
            ("update", "optim", "update"),
            ("adam", "adam", "adam"),
            ("dropout", "dropout", "dropout"),
        ];
        let pipelines = pipeline_entries
            .into_iter()
            .map(|(pipeline_name, shader_name, entry_point)| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(pipeline_name),
                    layout: None,
                    module: &shaders[shader_name],
                    entry_point,
                    compilation_options: Default::default(),
                    cache: None,
                });
                (pipeline_name, pipeline)
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

    pub fn zero_buffer(&self, len: usize) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor"),
                contents: bytemuck::cast_slice(&vec![0.0_f32; len.max(1)]),
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

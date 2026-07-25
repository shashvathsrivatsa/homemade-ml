// use crate::*;


// ——— Main ———————————————————————————————————————————————————————————————————————————————————————————————————————————

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub matmul_pipeline: wgpu::ComputePipeline,
}

impl Gpu {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap();
        println!("{:?}", adapter.get_info());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/matmul.wgsl").into()),
        });
        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });
        Self { device, queue, matmul_pipeline }
    }
}


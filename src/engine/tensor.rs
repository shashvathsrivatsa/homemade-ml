pub struct TensorNode {
    pub data: wgpu::Buffer,
    pub grad: wgpu::Buffer,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub len: usize,
    pub parents: Vec<usize>,
    pub op: &'static str,
}

impl TensorNode {
    pub(crate) fn from_buffers(data: wgpu::Buffer, grad: wgpu::Buffer, shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        let strides = (0..shape.len())
            .map(|dim| shape[dim + 1..].iter().product())
            .collect();
        Self {
            data,
            grad,
            shape,
            strides,
            len,
            parents: vec![],
            op: "",
        }
    }
}

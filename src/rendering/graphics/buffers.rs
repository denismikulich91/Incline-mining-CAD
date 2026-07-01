use super::*;

impl<'a> Graphics<'a> {
    pub(super) fn create_stream_buffer(
        device: &wgpu::Device,
        label: &'static str,
        size_bytes: usize,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (size_bytes.max(4)) as wgpu::BufferAddress,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(super) fn ensure_stream_capacity(
        device: &wgpu::Device,
        buffer: &mut wgpu::Buffer,
        capacity_items: &mut usize,
        required_items: usize,
        item_size: usize,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) {
        if required_items <= *capacity_items {
            return;
        }
        let new_capacity = required_items.next_power_of_two().max(1);
        *buffer = Self::create_stream_buffer(device, label, new_capacity * item_size, usage);
        *capacity_items = new_capacity;
    }

    pub(super) fn upload_scene_stream_buffers(&mut self) {
        if !self.lyon_buffer.vertices.is_empty() {
            Self::ensure_stream_capacity(
                &self.device,
                &mut self.lyon_vertex_gpu,
                &mut self.lyon_vertex_capacity,
                self.lyon_buffer.vertices.len(),
                std::mem::size_of::<Vertex>(),
                wgpu::BufferUsages::VERTEX,
                "Lyon Vertex Buffer",
            );
            self.queue.write_buffer(
                &self.lyon_vertex_gpu,
                0,
                bytemuck::cast_slice(&self.lyon_buffer.vertices),
            );
        }
        if !self.lyon_buffer.indices.is_empty() {
            Self::ensure_stream_capacity(
                &self.device,
                &mut self.lyon_index_gpu,
                &mut self.lyon_index_capacity,
                self.lyon_buffer.indices.len(),
                std::mem::size_of::<u32>(),
                wgpu::BufferUsages::INDEX,
                "Lyon Index Buffer",
            );
            self.queue.write_buffer(
                &self.lyon_index_gpu,
                0,
                bytemuck::cast_slice(&self.lyon_buffer.indices),
            );
        }
        if !self.stroke_vertex_buf.is_empty() {
            Self::ensure_stream_capacity(
                &self.device,
                &mut self.stroke_vertex_gpu,
                &mut self.stroke_vertex_capacity,
                self.stroke_vertex_buf.len(),
                std::mem::size_of::<StrokeVertex>(),
                wgpu::BufferUsages::VERTEX,
                "Stroke Vertex Buffer",
            );
            self.queue.write_buffer(
                &self.stroke_vertex_gpu,
                0,
                bytemuck::cast_slice(&self.stroke_vertex_buf),
            );
        }
        if !self.stroke_index_buf.is_empty() {
            Self::ensure_stream_capacity(
                &self.device,
                &mut self.stroke_index_gpu,
                &mut self.stroke_index_capacity,
                self.stroke_index_buf.len(),
                std::mem::size_of::<u32>(),
                wgpu::BufferUsages::INDEX,
                "Stroke Index Buffer",
            );
            self.queue.write_buffer(
                &self.stroke_index_gpu,
                0,
                bytemuck::cast_slice(&self.stroke_index_buf),
            );
        }
    }
}

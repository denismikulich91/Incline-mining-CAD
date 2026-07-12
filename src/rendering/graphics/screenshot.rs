//! One-shot viewport image export (File > Export Viewport Image...).
//!
//! The scene pass already resolves into an arbitrary target, so the capture
//! renders one extra frame into an offscreen texture (no egui chrome), copies
//! it into a mappable buffer alongside the normal frame's command encoder, and
//! encodes a PNG after the queue submit.

use std::path::PathBuf;

use super::*;
use crate::{userspace_log, userspace_warn};

/// GPU-side capture state produced before submit and consumed after present.
pub(super) struct PendingScreenshot {
    buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    path: PathBuf,
}

impl<'a> Graphics<'a> {
    /// Queue a viewport export; the next rendered frame writes the PNG.
    pub(crate) fn request_screenshot(&mut self, path: PathBuf) {
        self.pending_screenshot = Some(path);
        self.window.request_redraw();
    }

    /// Render the scene into an offscreen texture and record a copy of it into
    /// a mappable buffer on the frame's encoder. Runs before `queue.submit`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_screenshot_capture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        editor: &EditorState,
        triangulations: &[OpenTriangulation],
        block_models: &[OpenBlockModel],
        point_clouds: &[OpenPointCloud],
        rasters: &[OpenRasterTexture],
        path: PathBuf,
    ) -> PendingScreenshot {
        let width = self.config.width.max(1);
        let height = self.config.height.max(1);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Screenshot Target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_scene_pass(
            encoder,
            &view,
            editor,
            triangulations,
            block_models,
            point_clouds,
            rasters,
            true,
        );

        let padded_bytes_per_row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screenshot Readback Buffer"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        PendingScreenshot {
            buffer,
            padded_bytes_per_row,
            width,
            height,
            format: self.config.format,
            path,
        }
    }

    /// Map the readback buffer and write the PNG. Runs after `queue.submit`;
    /// blocks on the GPU, which is fine for a one-shot export.
    pub(super) fn finish_screenshot_capture(&self, capture: PendingScreenshot) {
        if let Err(error) = self.write_screenshot_png(&capture) {
            userspace_warn!(
                "Could not save viewport image {}: {error:#}",
                capture.path.display()
            );
        } else {
            userspace_log!("Saved viewport image: {}", capture.path.display());
        }
    }

    fn write_screenshot_png(&self, capture: &PendingScreenshot) -> Result<()> {
        // The egui pass writes raw bytes through a non-sRGB view of the same
        // texel layout, so both sRGB and non-sRGB variants read back the same
        // bytes; only the channel order matters here.
        let swap_bgra = match capture.format.remove_srgb_suffix() {
            wgpu::TextureFormat::Bgra8Unorm => true,
            wgpu::TextureFormat::Rgba8Unorm => false,
            other => {
                return Err(anyhow!(
                    "Unsupported surface format for image export: {other:?}"
                ));
            }
        };

        let (tx, rx) = std::sync::mpsc::channel();
        capture
            .buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = tx.send(result);
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| anyhow!("GPU poll failed: {error}"))?;
        rx.recv()
            .map_err(|_| anyhow!("Readback callback dropped"))?
            .map_err(|error| anyhow!("Buffer map failed: {error}"))?;

        let padded = capture.buffer.get_mapped_range(..);
        let row_bytes = capture.width as usize * 4;
        let mut rgba = Vec::with_capacity(row_bytes * capture.height as usize);
        for row in padded.chunks_exact(capture.padded_bytes_per_row as usize) {
            rgba.extend_from_slice(&row[..row_bytes]);
        }
        drop(padded);
        capture.buffer.unmap();

        if swap_bgra {
            for pixel in rgba.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        // The surface is opaque; alpha from the resolve can be anything.
        for pixel in rgba.chunks_exact_mut(4) {
            pixel[3] = 255;
        }

        let file = std::fs::File::create(&capture.path)?;
        let mut encoder =
            png::Encoder::new(std::io::BufWriter::new(file), capture.width, capture.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&rgba)?;
        writer.finish()?;
        Ok(())
    }
}

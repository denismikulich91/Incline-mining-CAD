use super::*;
use crate::userspace_log;

impl<'a> Graphics<'a> {
    pub(crate) async fn new(window: Arc<Window>) -> anyhow::Result<Graphics<'a>> {
        let window_size = window.inner_size();
        // Minimized/hidden Wayland windows may initially report 0×0. wgpu
        // surfaces, projection aspect ratios and attachments all require a
        // non-zero placeholder until the first real resize arrives.
        let size =
            winit::dpi::PhysicalSize::new(window_size.width.max(1), window_size.height.max(1));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| anyhow!("No compatible GPU adapter found: {e:?}"))?;

        let adapter_info = adapter.get_info();
        userspace_log!(
            "GPU Adapter: {} / {} / {:?} / {:?}",
            adapter_info.vendor,
            adapter_info.name,
            adapter_info.backend,
            adapter_info.device_type
        );
        userspace_log!(
            "GPU Driver: {} {}",
            adapter_info.driver,
            adapter_info.driver_info
        );

        let adapter_limits = adapter.limits();
        // Take everything the adapter offers for buffer size: large surfaces
        // can tessellate to multi-GiB vertex streams.
        let required_limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size,
            ..wgpu::Limits::default()
        };
        userspace_log!(
            "GPU Limits: max_buffer_size={} MiB, max_texture_dimension_2d={}, max_bind_groups={}",
            required_limits.max_buffer_size / (1024 * 1024),
            adapter_limits.max_texture_dimension_2d,
            adapter_limits.max_bind_groups
        );
        if required_limits.max_buffer_size < COMFORTABLE_MAX_BUFFER_SIZE {
            eprintln!(
                "GPU supports a maximum buffer size of {} MiB; large scenes may not display fully",
                required_limits.max_buffer_size / (1024 * 1024)
            );
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits,
                label: None,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        // wgpu treats uncaptured errors as fatal panics by default; a
        // validation failure (e.g. an oversized allocation) should degrade to
        // missing geometry, not lose the user's session.
        device.on_uncaptured_error(std::sync::Arc::new(|error: wgpu::Error| {
            crate::userspace_error!("wgpu error (continuing): {error}");
        }));

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .ok_or_else(|| anyhow!("Surface reports no sRGB texture format"))?;
        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|m| *m == wgpu::PresentMode::Fifo)
            .or_else(|| surface_caps.present_modes.first().copied())
            .ok_or_else(|| anyhow!("Surface reports no supported present modes"))?;
        let alpha_mode = surface_caps
            .alpha_modes
            .first()
            .copied()
            .ok_or_else(|| anyhow!("Surface reports no supported alpha modes"))?;
        // egui prefers a non-sRGB render target (it applies gamma itself), so
        // the GUI pass draws through a non-sRGB view of the sRGB surface.
        let gui_format = surface_format.remove_srgb_suffix();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode,
            view_formats: vec![gui_format],
            desired_maximum_frame_latency: 2,
        };
        let sample_count = MSAA_SAMPLE_COUNT;
        let (msaa_color, msaa_view) = Self::create_msaa_target(&device, &config, sample_count);
        let (depth_texture, depth_view) = Self::create_depth_target(&device, &config, sample_count);

        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/shader.wgsl"));
        let surface_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/surface.wgsl"));
        let block_model_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/block_model.wgsl"));
        let block_model_volume_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/block_model_volume.wgsl"));
        let block_model_transparency_fallback_shader = device.create_shader_module(
            wgpu::include_wgsl!("../shaders/block_model_transparency_fallback.wgsl"),
        );
        let block_model_transparency_composite_shader = device.create_shader_module(
            wgpu::include_wgsl!("../shaders/block_model_transparency_composite.wgsl"),
        );
        let block_model_volume_upscale_shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/block_model_volume_upscale.wgsl"
        ));
        let stroke_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/stroke.wgsl"));
        let edge_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/edge.wgsl"));
        let point_cloud_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/point_cloud.wgsl"));

        let camera = Camera::new(DVec3::new(0.0, 0.0, 10.0), (-90.0_f64).to_radians(), 0.0);
        let projection = Projection::new(
            config.width,
            config.height,
            INITIAL_CAMERA_Z_NEAR,
            INITIAL_CAMERA_Z_FAR,
        );
        let camera_controller = CameraController::new(0.6, 0.005, CAMERA_ROTATE_SENSITIVITY);
        let fly_camera_controller = FlyCameraController::new(232., CAMERA_ROTATE_SENSITIVITY);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection, DVec3::ZERO, 1.0);
        camera_uniform.update_viewport(config.width, config.height);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Some(&camera_bind_group_layout)],
                immediate_size: 0,
            });

        let style_bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let surface_style_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[style_bind_group_layout_entry],
                label: Some("surface_style_bind_group_layout"),
            });
        let surface_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Surface Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&camera_bind_group_layout),
                    Some(&surface_style_bind_group_layout),
                ],
                immediate_size: 0,
            });
        // Per-chunk rebase offset for triangulation surfaces (group 2); block
        // model pipelines keep the plain two-group surface layout above.
        let surface_chunk_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("surface_chunk_bind_group_layout"),
            });
        let raster_surface_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raster_surface_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let tri_surface_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Triangulation Surface Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&camera_bind_group_layout),
                    Some(&surface_style_bind_group_layout),
                    Some(&surface_chunk_bind_group_layout),
                    Some(&raster_surface_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let block_model_transparency_fallback_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                }],
                label: Some("block_model_transparency_fallback_bind_group_layout"),
            });
        let block_model_transparency_fallback_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Block Model Transparency Fallback Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&camera_bind_group_layout),
                    Some(&surface_style_bind_group_layout),
                    Some(&block_model_transparency_fallback_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let block_model_transparency_composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
                label: Some("block_model_transparency_composite_bind_group_layout"),
            });
        let block_model_transparency_composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Block Model Transparency Composite Pipeline Layout"),
                bind_group_layouts: &[Some(&block_model_transparency_composite_bind_group_layout)],
                immediate_size: 0,
            });
        let block_model_volume_upscale_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("block_model_volume_upscale_bind_group_layout"),
            });
        let block_model_volume_upscale_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Block Model Volume Upscale Pipeline Layout"),
                bind_group_layouts: &[Some(&block_model_volume_upscale_bind_group_layout)],
                immediate_size: 0,
            });
        let block_model_volume_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("block_model_volume_bind_group_layout"),
            });
        let block_model_volume_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Block Model Volume Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&camera_bind_group_layout),
                    Some(&block_model_volume_bind_group_layout),
                    Some(&block_model_transparency_fallback_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let edge_style_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[style_bind_group_layout_entry],
                label: Some("edge_style_bind_group_layout"),
            });
        let edge_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Edge Pipeline Layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&edge_style_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
        }];
        let surface_vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SurfaceVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        }];
        // One instance per block: lower.xyz + grade, then upper.xyz + pad.
        // The shader expands vertex_index 0..36 into the cube's faces.
        let block_model_vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BlockInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x3],
        }];

        let stroke_vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<StrokeVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32x3, 3 => Float32x2, 4 => Float32],
        }];
        let edge_instance_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<EdgeInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        }];
        let point_uncolored_instance_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PointPosition>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        }];
        let point_colored_instance_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PointInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Unorm8x4],
        }];

        let create_stroke_pipeline = |label, depth_stencil| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &stroke_shader,
                    entry_point: Some("vs_main"),
                    buffers: &stroke_vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &stroke_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let stroke_render_pipeline = create_stroke_pipeline(
            "Depth-tested Stroke Render Pipeline",
            Some(Self::depth_state(false, -1)),
        );
        let opaque_stroke_render_pipeline = create_stroke_pipeline(
            "Opaque Depth-writing Stroke Render Pipeline",
            Some(Self::depth_state(true, -1)),
        );
        let edge_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Instanced Triangulation Edge Pipeline"),
            layout: Some(&edge_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &edge_shader,
                entry_point: Some("vs_main"),
                buffers: &edge_instance_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &edge_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(Self::depth_state(false, -1)),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        // Point splats reuse the edge pipeline layout (camera + one style
        // uniform) but write depth so clouds occlude correctly against
        // meshes and themselves.
        let point_cloud_colored_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Colored Point Cloud Pipeline"),
                layout: Some(&edge_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &point_cloud_shader,
                    entry_point: Some("vs_colored"),
                    buffers: &point_colored_instance_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &point_cloud_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(Self::depth_state(true, 0)),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        let point_cloud_uncolored_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Uncolored Point Cloud Pipeline"),
                layout: Some(&edge_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &point_cloud_shader,
                    entry_point: Some("vs_uncolored"),
                    buffers: &point_uncolored_instance_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &point_cloud_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(Self::depth_state(true, 0)),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        let mut overlay_depth = Self::depth_state(false, 0);
        overlay_depth.depth_compare = Some(wgpu::CompareFunction::Always);
        let overlay_render_pipeline =
            create_stroke_pipeline("Editor Overlay Render Pipeline", Some(overlay_depth));

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(Self::depth_state(true, 0)),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        // Triangulation surface pipelines use position-only vertices with a per-draw colour
        // uniform.
        let create_tri_surface_pipeline = |label, write_depth, depth_compare| {
            let mut depth = Self::depth_state(write_depth, 0);
            depth.depth_compare = Some(depth_compare);
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&tri_surface_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &surface_shader,
                    entry_point: Some("vs_main"),
                    buffers: &surface_vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &surface_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(depth),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let surface_render_pipeline = create_tri_surface_pipeline(
            "Opaque Triangulation Surface Pipeline",
            true,
            wgpu::CompareFunction::GreaterEqual,
        );
        let transparent_surface_render_pipeline = create_tri_surface_pipeline(
            "Transparent Triangulation Surface Pipeline",
            false,
            wgpu::CompareFunction::GreaterEqual,
        );
        // Flat plan-view images for undraped rasters: drawn first, pinned to
        // the far plane, no depth writes, so all scene geometry covers them.
        let raster_plane_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/raster_plane.wgsl"));
        let raster_plane_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Raster Plane Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&camera_bind_group_layout),
                    Some(&raster_surface_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let raster_plane_vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: (std::mem::size_of::<f32>() * 4) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
        }];
        let raster_plane_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Raster Plane Pipeline"),
                layout: Some(&raster_plane_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &raster_plane_shader,
                    entry_point: Some("vs_main"),
                    buffers: &raster_plane_vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &raster_plane_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(Self::depth_state(false, 0)),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        let create_block_model_surface_pipeline = |label, write_depth, depth_compare| {
            let mut depth = Self::depth_state(write_depth, 0);
            depth.depth_compare = Some(depth_compare);
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&surface_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &block_model_shader,
                    entry_point: Some("vs_main"),
                    buffers: &block_model_vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &block_model_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(depth),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let block_model_render_pipeline = create_block_model_surface_pipeline(
            "Opaque Block Model Surface Pipeline",
            true,
            wgpu::CompareFunction::GreaterEqual,
        );
        let block_model_volume_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Block Model Volume Raycast Pipeline"),
                layout: Some(&block_model_volume_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &block_model_volume_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &block_model_volume_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                // The raycast now renders into the single-sample off-screen
                // volume target (upscaled afterwards), not the MSAA surface, so
                // this must be 1 to match the attachment.
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        let block_model_transparency_fallback_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Block Model Transparency Fallback Pipeline"),
                layout: Some(&block_model_transparency_fallback_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &block_model_transparency_fallback_shader,
                    entry_point: Some("vs_main"),
                    buffers: &block_model_vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &block_model_transparency_fallback_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        let block_model_transparency_composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Block Model Transparency Composite Pipeline"),
                layout: Some(&block_model_transparency_composite_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &block_model_transparency_composite_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &block_model_transparency_composite_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        let block_model_volume_upscale_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Block Model Volume Upscale Pipeline"),
                layout: Some(&block_model_volume_upscale_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &block_model_volume_upscale_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &block_model_volume_upscale_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        // Premultiplied over: matches the direct volume pass
                        // this replaces.
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        // Document-object xray pipeline keeps position+colour per-vertex.
        let create_surface_pipeline =
            |label, write_depth, depth_compare, shader_module: &wgpu::ShaderModule| {
                let mut depth = Self::depth_state(write_depth, 0);
                depth.depth_compare = Some(depth_compare);
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: shader_module,
                        entry_point: Some("vs_main"),
                        buffers: &vertex_buffers,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader_module,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: config.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(depth),
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview_mask: None,
                    cache: None,
                })
            };
        let xray_render_pipeline = create_surface_pipeline(
            "X-Ray Document Fill Pipeline",
            false,
            wgpu::CompareFunction::Always,
            &shader,
        );
        let transparent_document_fill_pipeline = create_surface_pipeline(
            "Transparent Document Fill Pipeline",
            false,
            wgpu::CompareFunction::GreaterEqual,
            &shader,
        );

        let lyon_buffer: VertexBuffers<Vertex, u32> = VertexBuffers::new();
        let lyon_vertex_gpu = Self::create_stream_buffer(
            &device,
            "Lyon Vertex Buffer",
            std::mem::size_of::<Vertex>(),
            wgpu::BufferUsages::VERTEX,
        );
        let lyon_index_gpu = Self::create_stream_buffer(
            &device,
            "Lyon Index Buffer",
            std::mem::size_of::<u32>(),
            wgpu::BufferUsages::INDEX,
        );
        let stroke_vertex_gpu = Self::create_stream_buffer(
            &device,
            "Stroke Vertex Buffer",
            std::mem::size_of::<StrokeVertex>(),
            wgpu::BufferUsages::VERTEX,
        );
        let stroke_index_gpu = Self::create_stream_buffer(
            &device,
            "Stroke Index Buffer",
            std::mem::size_of::<u32>(),
            wgpu::BufferUsages::INDEX,
        );
        let overlay_vertex_gpu = Self::create_stream_buffer(
            &device,
            "Editor Overlay Vertex Buffer",
            std::mem::size_of::<StrokeVertex>(),
            wgpu::BufferUsages::VERTEX,
        );
        let overlay_index_gpu = Self::create_stream_buffer(
            &device,
            "Editor Overlay Index Buffer",
            std::mem::size_of::<u32>(),
            wgpu::BufferUsages::INDEX,
        );
        let dynamic_vertex_gpu = Self::create_stream_buffer(
            &device,
            "Dynamic Scene Vertex Buffer",
            std::mem::size_of::<StrokeVertex>(),
            wgpu::BufferUsages::VERTEX,
        );
        let dynamic_index_gpu = Self::create_stream_buffer(
            &device,
            "Dynamic Scene Index Buffer",
            std::mem::size_of::<u32>(),
            wgpu::BufferUsages::INDEX,
        );

        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &cache);
        let mut text_atlas = TextAtlas::new(&device, &queue, &cache, surface_format);
        // The world-space glyphon fork emits transformed text at clip depth
        // zero. With our reversed-Z depth buffer that puts it behind every
        // triangulation pixel, so document labels must use the overlay depth
        // mode until glyphon preserves their projected world depth.
        let mut text_depth = Self::depth_state(false, -2);
        text_depth.depth_compare = Some(wgpu::CompareFunction::Always);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            Some(text_depth),
        );
        let text_cache = TextCache::new();
        let text_system = TextSystem {
            font_system,
            swash_cache,
            text_renderer,
            text_atlas,
            text_cache,
            viewport,
        };
        let gui = Gui::new(&window, &device, gui_format);
        // High-precision block-model attachments are created on first visible
        // use; ordinary document and topology views pay no full-screen VRAM
        // cost for them.
        let block_model_transparency_targets = None;
        let block_model_volume_target = None;
        Ok(Self {
            gui,
            text_system,
            surface_render_pipeline,
            transparent_surface_render_pipeline,
            raster_plane_render_pipeline,
            block_model_render_pipeline,
            block_model_volume_pipeline,
            block_model_transparency_fallback_pipeline,
            block_model_transparency_composite_pipeline,
            block_model_volume_upscale_pipeline,
            block_model_volume_upscale_bind_group_layout,
            block_model_transparency_fallback_bind_group_layout,
            block_model_transparency_composite_bind_group_layout,
            block_model_volume_bind_group_layout,
            surface_style_bind_group_layout,
            surface_chunk_bind_group_layout,
            raster_surface_bind_group_layout,
            render_pipeline,
            transparent_document_fill_pipeline,
            xray_render_pipeline,
            opaque_stroke_render_pipeline,
            stroke_render_pipeline,
            edge_render_pipeline,
            point_cloud_colored_render_pipeline,
            point_cloud_uncolored_render_pipeline,
            edge_style_bind_group_layout,
            overlay_render_pipeline,
            lyon_vertex_gpu,
            lyon_index_gpu,
            stroke_vertex_gpu,
            stroke_index_gpu,
            overlay_vertex_gpu,
            overlay_index_gpu,
            dynamic_vertex_gpu,
            dynamic_index_gpu,
            camera_buffer,
            camera_bind_group,
            msaa_color,
            msaa_view,
            depth_texture,
            depth_view,
            block_model_transparency_targets,
            block_model_volume_target,
            instance,
            adapter,
            window,
            surface,
            queue,
            device,
            config,
            sample_count,
            size,
            lyon_buffer,
            lyon_vertex_capacity: 1,
            lyon_index_capacity: 1,
            camera,
            camera_uniform,
            camera_controller,
            fly_camera_controller,
            projection,
            mouse_pressed: None,
            fly_mode_enabled: false,
            slice_view: None,
            stroke_index_buf: Vec::new(),
            stroke_vertex_buf: Vec::new(),
            stroke_vertex_capacity: 1,
            stroke_index_capacity: 1,
            overlay_vertex_buf: Vec::new(),
            overlay_index_buf: Vec::new(),
            overlay_vertex_capacity: 1,
            overlay_index_capacity: 1,
            dynamic_vertex_buf: Vec::new(),
            dynamic_index_buf: Vec::new(),
            dynamic_vertex_capacity: 1,
            dynamic_index_capacity: 1,
            cached_textareas: Vec::new(),
            textarea_depths: Vec::new(),
            text_prepare_pending: true,
            frame_index: 0,
            last_text_cache_trim_frame: 0,
            last_interaction: None,
            geometry_dirty: true,
            cached_document_revision: u64::MAX,
            cached_render_style_key: None,
            cached_bounds_document_revision: u64::MAX,
            cached_scene_bounds: None,
            overlay_dirty: true,
            cached_scale_factor: 0.0,
            cached_measurement_state: (false, None, None, Vec::new()),
            cached_poly_finish_dialog: false,
            pick_records: Vec::new(),
            text_pick_records: Vec::new(),
            document_draw_batches: Vec::new(),
            orbit_marker: None,
            scene_origin: DVec3::ZERO,
            vertical_exaggeration: 1.0,
            triangulation_gpu: TriangulationGpuCache::default(),
            static_strokes: StaticStrokeCache::default(),
            block_model_gpu: BlockModelGpuCache::default(),
            point_cloud_gpu: PointCloudGpuCache::default(),
            raster_gpu: RasterGpuCache::default(),
            chunk_render_stats: (0, 0),
            pending_screenshot: None,
            slice_preview: None,
            embedded_slice_preview: None,
            embedded_preview_scene_key: None,
            detached_preview_scene_key: None,
        })
    }

    pub(crate) fn reconfigure(&mut self) {
        self.resize(self.size);
    }

    pub(crate) fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.mark_interaction();
            // Screen-space (identity-transform) text bakes the resolution
            // into its vertices during prepare.
            self.text_prepare_pending = true;
            self.projection.resize(new_size.width, new_size.height);
            self.camera_uniform
                .update_viewport(new_size.width, new_size.height);
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.text_system.viewport.update(
                &self.queue,
                Resolution {
                    width: new_size.width,
                    height: new_size.height,
                },
                glyphon::CameraUniform {
                    view_proj: self.camera_uniform.view_proj,
                },
            );
            self.surface.configure(&self.device, &self.config);
            let (msaa_color, msaa_view) =
                Self::create_msaa_target(&self.device, &self.config, self.sample_count);
            self.msaa_color = msaa_color;
            self.msaa_view = msaa_view;
            let (depth_texture, depth_view) =
                Self::create_depth_target(&self.device, &self.config, self.sample_count);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
            // Any lazily-created attachments refer to the old size/depth
            // view. Drop them now and recreate only if the resized viewport
            // actually renders a block model.
            self.block_model_transparency_targets = None;
            self.block_model_volume_target = None;
            // Document geometry is stored in world space and screen-space stroke
            // sizing is handled by the viewport uniform. Resizing therefore only
            // requires new surface-sized attachments; rebuilding and re-uploading
            // the entire document here makes interactive resize needlessly laggy.
            self.overlay_dirty = true;
        }
    }
}

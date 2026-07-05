use super::*;
use crate::userspace_log;

impl<'a> Graphics<'a> {
    pub(crate) async fn new(window: Arc<Window>) -> anyhow::Result<Graphics<'a>> {
        let size = window.inner_size();

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
                power_preference: wgpu::PowerPreference::default(),
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
        let required_limits = wgpu::Limits {
            max_buffer_size: REQUESTED_MAX_BUFFER_SIZE.min(adapter_limits.max_buffer_size),
            ..wgpu::Limits::default()
        };
        userspace_log!(
            "GPU Limits: max_buffer_size={} MiB, max_texture_dimension_2d={}, max_bind_groups={}",
            required_limits.max_buffer_size / (1024 * 1024),
            adapter_limits.max_texture_dimension_2d,
            adapter_limits.max_bind_groups
        );
        if required_limits.max_buffer_size < REQUESTED_MAX_BUFFER_SIZE {
            eprintln!(
                "GPU supports a maximum buffer size of {} MiB; the requested 2048 MiB is unavailable",
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

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| surface_caps.formats.first().copied())
            .ok_or_else(|| anyhow!("Surface reports no supported texture formats"))?;
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
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode,
            view_formats: vec![],
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
        let block_model_peel_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/block_model_peel.wgsl"));
        let peel_composite_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/peel_composite.wgsl"));
        let block_model_volume_upscale_shader = device.create_shader_module(wgpu::include_wgsl!(
            "../shaders/block_model_volume_upscale.wgsl"
        ));
        let stroke_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/stroke.wgsl"));
        let edge_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/edge.wgsl"));

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
        let block_model_peel_bind_group_layout =
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
                label: Some("block_model_peel_bind_group_layout"),
            });
        let block_model_peel_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Block Model Depth Peel Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&camera_bind_group_layout),
                    Some(&surface_style_bind_group_layout),
                    Some(&block_model_peel_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let block_model_peel_composite_bind_group_layout =
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
                label: Some("block_model_peel_composite_bind_group_layout"),
            });
        let block_model_peel_composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Block Model Peel Composite Pipeline Layout"),
                bind_group_layouts: &[Some(&block_model_peel_composite_bind_group_layout)],
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
                    Some(&block_model_peel_bind_group_layout),
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
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
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
                layout: Some(&surface_pipeline_layout),
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
            wgpu::CompareFunction::LessEqual,
        );
        let transparent_surface_render_pipeline = create_tri_surface_pipeline(
            "Transparent Triangulation Surface Pipeline",
            false,
            wgpu::CompareFunction::LessEqual,
        );
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
            wgpu::CompareFunction::LessEqual,
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
        let block_model_peel_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Block Model Depth Peel Pipeline"),
                layout: Some(&block_model_peel_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &block_model_peel_shader,
                    entry_point: Some("vs_main"),
                    buffers: &block_model_vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &block_model_peel_shader,
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
        let block_model_peel_composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Block Model Peel Composite Pipeline"),
                layout: Some(&block_model_peel_composite_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &peel_composite_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &peel_composite_shader,
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
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            Some(Self::depth_state(false, -2)),
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
        let gui = Gui::new(&window, &device, surface_format);
        let block_model_peel_targets = Self::create_block_model_peel_targets(
            &device,
            &config,
            &depth_view,
            &block_model_peel_bind_group_layout,
            &block_model_peel_composite_bind_group_layout,
        );
        let block_model_volume_target = Self::create_block_model_volume_target(
            &device,
            &config,
            &block_model_volume_upscale_bind_group_layout,
        );
        Ok(Self {
            gui,
            text_system,
            surface_render_pipeline,
            transparent_surface_render_pipeline,
            block_model_render_pipeline,
            block_model_volume_pipeline,
            block_model_peel_pipeline,
            block_model_peel_composite_pipeline,
            block_model_volume_upscale_pipeline,
            block_model_volume_upscale_bind_group_layout,
            block_model_peel_bind_group_layout,
            block_model_peel_composite_bind_group_layout,
            block_model_volume_bind_group_layout,
            surface_style_bind_group_layout,
            render_pipeline,
            xray_render_pipeline,
            stroke_render_pipeline,
            edge_render_pipeline,
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
            block_model_peel_targets,
            block_model_volume_target,
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
            last_interaction: None,
            geometry_dirty: true,
            cached_document_revision: u64::MAX,
            cached_bounds_document_revision: u64::MAX,
            cached_scene_bounds: None,
            overlay_dirty: true,
            cached_scale_factor: 0.0,
            cached_measurement_state: (false, None, None),
            cached_poly_finish_dialog: false,
            pick_records: Vec::new(),
            text_pick_records: Vec::new(),
            orbit_marker: None,
            scene_origin: DVec3::ZERO,
            vertical_exaggeration: 1.0,
            triangulation_gpu: TriangulationGpuCache::default(),
            block_model_gpu: BlockModelGpuCache::default(),
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
            self.block_model_peel_targets = Self::create_block_model_peel_targets(
                &self.device,
                &self.config,
                &self.depth_view,
                &self.block_model_peel_bind_group_layout,
                &self.block_model_peel_composite_bind_group_layout,
            );
            self.block_model_volume_target = Self::create_block_model_volume_target(
                &self.device,
                &self.config,
                &self.block_model_volume_upscale_bind_group_layout,
            );
            // Document geometry is stored in world space and screen-space stroke
            // sizing is handled by the viewport uniform. Resizing therefore only
            // requires new surface-sized attachments; rebuilding and re-uploading
            // the entire document here makes interactive resize needlessly laggy.
            self.overlay_dirty = true;
        }
    }
}

#[cfg(test)]
mod shader_tests {
    /// Parse + validate a WGSL source with all capabilities enabled, so the
    /// only failures are genuine syntax/type errors — the class of bug that
    /// otherwise only surfaces at runtime pipeline creation on a real GPU.
    fn validate(name: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|error| {
            panic!("{name} failed to parse:\n{}", error.emit_to_string(source))
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name} failed to validate: {error:?}"));
    }

    #[test]
    fn all_shaders_parse_and_validate() {
        validate("shader.wgsl", include_str!("../shaders/shader.wgsl"));
        validate("surface.wgsl", include_str!("../shaders/surface.wgsl"));
        validate(
            "block_model.wgsl",
            include_str!("../shaders/block_model.wgsl"),
        );
        validate(
            "block_model_volume.wgsl",
            include_str!("../shaders/block_model_volume.wgsl"),
        );
        validate(
            "block_model_peel.wgsl",
            include_str!("../shaders/block_model_peel.wgsl"),
        );
        validate(
            "peel_composite.wgsl",
            include_str!("../shaders/peel_composite.wgsl"),
        );
        validate(
            "block_model_volume_upscale.wgsl",
            include_str!("../shaders/block_model_volume_upscale.wgsl"),
        );
        validate("stroke.wgsl", include_str!("../shaders/stroke.wgsl"));
        validate("edge.wgsl", include_str!("../shaders/edge.wgsl"));
    }
}

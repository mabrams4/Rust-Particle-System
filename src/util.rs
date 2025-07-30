use bevy::{
    prelude::*,
    render::{
        render_resource::*, 
        renderer::RenderDevice,
        view::Msaa,
    },
};

pub fn get_bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout
{
    // create the bind group layout
    render_device.create_bind_group_layout(
        "bind_group_layout",
        &[BindGroupLayoutEntry
        {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None
        },
        BindGroupLayoutEntry
        {
            binding: 1,
            visibility: ShaderStages::VERTEX | ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None
        }
        ]
    )
}

pub fn get_render_pipeline_descriptor(
    bind_group_layout: &BindGroupLayout,
    shader_handle: &Handle<Shader>) -> RenderPipelineDescriptor
{
    RenderPipelineDescriptor 
    {   label: Some("render_pipeline_descriptor".into()), 
        layout: vec![bind_group_layout.clone()], 
        push_constant_ranges: vec![], 
        vertex: VertexState
        {
            shader: shader_handle.clone(),
            shader_defs: vec![],
            entry_point: "vertex_main".into(),
            buffers: vec![
                VertexBufferLayout {
                    array_stride: 16, // 8 bytes per vec2, two vec2s = 16
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![
                        VertexAttribute {
                            shader_location: 0,
                            offset: 0,
                            format: VertexFormat::Float32x2, // position
                        },
                        VertexAttribute {
                            shader_location: 1,
                            offset: 8,
                            format: VertexFormat::Float32x2, // uv
                        },
                    ],
                }
            ]
        }, 
        primitive: PrimitiveState 
        {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None, 
        multisample: MultisampleState
        {
            count: Msaa::Sample4 as u32,
            mask: !0,
            alpha_to_coverage_enabled: false
        },
        fragment: Some(FragmentState
        {
            shader: shader_handle.clone(),
            shader_defs: vec![],
            entry_point: "fragment_main".into(),
            targets: vec![Some(ColorTargetState 
                {
                format: TextureFormat::Rgba8UnormSrgb,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
                })]
        }), 
        zero_initialize_workgroup_memory: false 
    }
}

pub fn get_compute_pipeline_descriptor(
    bind_group_layout: &BindGroupLayout,
    shader_handle: &Handle<Shader>) -> ComputePipelineDescriptor
{
    ComputePipelineDescriptor 
    {   
        label: Some("compute_pipeline_id".into()), 
        layout: vec![bind_group_layout.clone()],
        push_constant_ranges: vec![], 
        shader: shader_handle.clone(), 
        shader_defs: vec![], 
        entry_point: "compute_main".into(), 
        zero_initialize_workgroup_memory: false 
    }
}
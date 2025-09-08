use bevy::{
    prelude::*,
    render::{
        render_resource::*, 
        renderer::RenderDevice,
        view::Msaa,
    },
};
use std::borrow::Cow;
use std::num::NonZeroU64;
use crate::particle_render::SortingParams;

// returns the bind group layout for group 0 (used by render shader and main compute shader)
pub fn get_bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout
{
    // create the bind group layout
    render_device.create_bind_group_layout(
        "bind_group_layout0",
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
        },
        BindGroupLayoutEntry
        {
            binding: 2,
            visibility: ShaderStages::VERTEX | ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: NonZeroU64::new(std::mem::size_of::<SortingParams>() as u64),
            },
            count: None
        },
        BindGroupLayoutEntry
        {
            binding: 3,
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
            binding: 4,
            visibility: ShaderStages::VERTEX | ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None
        },
        ]
    )
}

// returns bind group for group 0 
pub fn get_bind_group(
    label: &str,
    render_device: &RenderDevice,
    bind_group_layout: &BindGroupLayout,
    particle_buffer: &Buffer,
    particle_buffer_size: std::num::NonZeroU64,
    config_buffer: &Buffer,
    config_buffer_size: std::num::NonZeroU64,
    spatial_lookup_buffer: &Buffer,
    spatial_lookup_buffer_size: std::num::NonZeroU64,
    grid_start_idxs_buffer: &Buffer,
    grid_start_idxs_buffer_size: std::num::NonZeroU64,
    sorting_params_buffer: &Buffer,
) -> BindGroup
{
    render_device.create_bind_group(
    label, 
    bind_group_layout, 
    &[
        BindGroupEntry 
        {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding 
                {   
                    buffer: &particle_buffer, 
                    offset: 0, 
                    size: Some(particle_buffer_size)
                })
        },
        BindGroupEntry 
        {
            binding: 1,
            resource: BindingResource::Buffer(BufferBinding 
                {   
                    buffer: &config_buffer, 
                    offset: 0, 
                    size: Some(config_buffer_size)
                })
        },
        BindGroupEntry
        {
            binding: 2,
            resource: BindingResource::Buffer(BufferBinding 
                {   
                    buffer: &sorting_params_buffer, 
                    offset: 0, 
                    size: NonZeroU64::new(std::mem::size_of::<SortingParams>() as u64)
                })
        },
        BindGroupEntry
        {
            binding: 3,
            resource: BindingResource::Buffer(BufferBinding 
                {   
                    buffer: &spatial_lookup_buffer, 
                    offset: 0, 
                    size: Some(spatial_lookup_buffer_size)
                })
        },
        BindGroupEntry
        {
            binding: 4,
            resource: BindingResource::Buffer(BufferBinding 
                {   
                    buffer: &grid_start_idxs_buffer, 
                    offset: 0, 
                    size: Some(grid_start_idxs_buffer_size)
                })
        }
    ])
}

// returns pipeline descriptor for render pipeline
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

// returns pipeline descriptor for compute pipeline
pub fn get_compute_pipeline_descriptor(
    bind_group_layout: &BindGroupLayout,
    shader_handle: &Handle<Shader>,
    entry_point: &str,
) -> ComputePipelineDescriptor
{
    ComputePipelineDescriptor 
    {   
        label: None, 
        layout: vec![bind_group_layout.clone()],
        push_constant_ranges: vec![], 
        shader: shader_handle.clone(), 
        shader_defs: vec![], 
        entry_point: Cow::from(entry_point.to_owned()), 
        zero_initialize_workgroup_memory: false 
    }
}
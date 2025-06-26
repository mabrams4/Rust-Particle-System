use bevy::{
    prelude::*,
    render::{
        render_graph::{self, Node, RenderGraphContext, RenderLabel}, 
        render_resource::*, 
        renderer::{RenderContext, RenderDevice},
        view::{ExtractedView, ViewTarget, Msaa},
    },
};

use crate::{particle_render::render_graph::NodeRunError, ParticleConfig};
use crate::ParticleSystem;
use crate::particle::Particle;

#[derive(RenderLabel, Hash, Debug, Eq, PartialEq, Clone)]
pub struct ParticleRenderLabel;

#[derive(Component)]
pub struct PreparedParticleBuffer {
    pub bind_group: BindGroup,
    pub num_particles: u32,
    pub vertex_buffer: Buffer
}

#[derive(Resource)]
pub struct ParticleRenderPipeline 
{
    bind_group_layout: BindGroupLayout,
    render_pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for ParticleRenderPipeline 
{
    // called when Pipeline is created
    fn from_world(world: &mut World) -> Self 
    {
        // get render device
        let render_device = world.resource::<RenderDevice>();

        // get shader handle
        let shader_handle = world.resource::<AssetServer>().load("render_shader.wgsl");
        
        // create the bind group layout
        let bind_group_layout = render_device.create_bind_group_layout(
            "bind_group_layout",
            &[BindGroupLayoutEntry
            {
                binding: 0,
                visibility: ShaderStages::VERTEX,
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
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None
            }
            ]
        );

        info!("[S-Render] created bind group layout");

        // create the render pipeline and store it in the pipeline cache
        let pipeline_cache = world.resource_mut::<PipelineCache>();

        let render_pipeline_id = pipeline_cache.queue_render_pipeline(
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
                    shader: shader_handle,
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
        );
        info!("[S] created render pipeline");
        // return the ParticleRenderPipeline object
        ParticleRenderPipeline 
        {  
            bind_group_layout,
            render_pipeline_id
        }
    }
}

pub struct ParticleRenderNode 
{
    //view_query: QueryState<(&'static ExtractedView, &'static ViewTarget)>,
    view_query: QueryState<&'static ViewTarget>,
    particle_system: QueryState<Entity, With<ParticleSystem>>,
}

impl Node for ParticleRenderNode 
{
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> 
    {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ParticleRenderPipeline>();
        for target in self.view_query.iter_manual(world) 
        {
            for entity in self.particle_system.iter_manual(world)
            {
                // check if pipeline is ready yet
                if let Some(pipeline_id) = pipeline_cache.get_render_pipeline(pipeline.render_pipeline_id)
                {
                    //let particle_system = world.get::<ParticleSystem>(entity).unwrap();
                    let prepared_particle_buffer = world.get::<PreparedParticleBuffer>(entity).unwrap();

                    let mut render_pass = RenderContext::begin_tracked_render_pass(
                        render_context, 
                        RenderPassDescriptor
                        {
                            label: Some("render_pass_descriptor"),
                            color_attachments: &[Some(target.get_color_attachment())],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None
                        }
                    );

                    render_pass.set_render_pipeline(pipeline_id);
                    render_pass.set_bind_group(0, &prepared_particle_buffer.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, prepared_particle_buffer.vertex_buffer.slice(..));
                    render_pass.draw(0..6, 0..prepared_particle_buffer.num_particles as u32);
                }
            }
        }
        Ok(())
    }

    fn update(&mut self, world: &mut World) {
        self.particle_system.update_archetypes(world);
        self.view_query.update_archetypes(world);
    }
}

impl ParticleRenderNode {
    pub fn new(world: &mut World) -> Self 
    {
        Self 
        {
            view_query: QueryState::new(world),
            particle_system: QueryState::new(world),
        }
    }
}

pub fn prepare_particles(
    render_device: Res<RenderDevice>,
    particle_system_query: Query<(Entity, &ParticleSystem)>,
    render_pipeline: Res<ParticleRenderPipeline>,
    mut config: ResMut<ParticleConfig>,
    camera_query: Query<&ExtractedView, With<Camera>>,
    time: Res<Time>,
    mut commands: Commands
)
{
    if let Ok(view) = camera_query.single() {
        let view_matrix = view.world_from_view.compute_matrix().inverse();
        let view_proj = view.clip_from_view * view_matrix;
        config.view_proj = view_proj.to_cols_array_2d();
    }
    config.delta_time = time.delta().as_secs_f32();

    let uniform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor 
        {   label: Some("uniform_buffer"), 
            contents: bytemuck::bytes_of(config.as_ref()),
            usage: BufferUsages::UNIFORM 
        });
    let uniform_buffer_size = uniform_buffer.size();
    let uniform_buffer_size = std::num::NonZeroU64::new(uniform_buffer_size).unwrap();

    if let Ok((entity, particle_system)) = particle_system_query.single() 
    {
        let particles = &particle_system.particles;

        let mut byte_buffer = Vec::<u8>::new();
        let mut buffer = encase::StorageBuffer::new(&mut byte_buffer);
        buffer.write(&particles).unwrap();

        let storage_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor 
            {   label: Some("storage_buffer"), 
                contents: buffer.into_inner(), 
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST
            });
        let storage_buffer_size = (std::mem::size_of::<Particle>() * config.particle_count as usize) as u64;
        let storage_buffer_size = std::num::NonZeroU64::new(storage_buffer_size).unwrap();

        let bind_group = render_device.create_bind_group(
            "bind_group", 
            &render_pipeline.bind_group_layout, 
            &[
                BindGroupEntry 
                {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding 
                        {   
                            buffer: &storage_buffer, 
                            offset: 0, 
                            size: Some(storage_buffer_size)
                        })
                },
                BindGroupEntry 
                {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding 
                        {   
                            buffer: &uniform_buffer, 
                            offset: 0, 
                            size: Some(uniform_buffer_size)
                        })
                }
            ]);

        let quad_vertices: &[f32; 24] = &[
            // x,    y,    u,    v
            -0.5, -0.5, 0.0, 1.0, // bottom-left
            0.5, -0.5, 1.0, 1.0, // bottom-right
            -0.5,  0.5, 0.0, 0.0, // top-left
            0.5, -0.5, 1.0, 1.0, // bottom-right
            0.5,  0.5, 1.0, 0.0, // top-right
            -0.5,  0.5, 0.0, 0.0, // top-left
        ];

        // Upload to GPU:
        let vertex_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("quad_vertex_buffer"),
            contents: bytemuck::cast_slice(quad_vertices),
            usage: BufferUsages::VERTEX,
        });
        
        commands.entity(entity).insert(PreparedParticleBuffer 
            {
                bind_group: bind_group,
                num_particles: config.particle_count,
                vertex_buffer: vertex_buffer,
            });
    }
}
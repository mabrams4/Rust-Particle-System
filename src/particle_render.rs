use bevy::{
    prelude::*,
    render::{
        render_graph::{self, Node, RenderGraphContext, RenderLabel}, 
        render_resource::{*}, 
        renderer::{RenderContext, RenderDevice, RenderQueue},
        view::{ExtractedView, ViewTarget},
    },
};

use crate::{particle_render::render_graph::NodeRunError, ParticleConfig};
use crate::ParticleSystem;
use crate::particle::Particle;
use crate::util::{get_bind_group_layout, get_render_pipeline_descriptor};

#[derive(RenderLabel, Hash, Debug, Eq, PartialEq, Clone)]
pub struct ParticleRenderLabel;

#[derive(Component)]
pub struct PreparedParticleBuffer {
    pub num_particles: u32,
    pub bind_group: BindGroup,  // shared between vertex and compute shaders
    pub vertex_buffer: Buffer,
    pub storage_buffer: Buffer,
    pub uniform_buffer: Buffer,
}

#[derive(Resource)]
pub struct ParticleRenderPipeline 
{
    bind_group_layout: BindGroupLayout, // shared with compute shader
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
        
        let bind_group_layout = get_bind_group_layout(render_device);

        // create the render pipeline and store it in the pipeline cache
        let pipeline_cache = world.resource_mut::<PipelineCache>();

        let render_pipeline_id = pipeline_cache.queue_render_pipeline(
            get_render_pipeline_descriptor(&bind_group_layout, &shader_handle)
        );
        info!("[Setup] created render pipeline");
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
                    if let Some(prepared_particle_buffer) = world.get::<PreparedParticleBuffer>(entity)
                    {
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
    render_queue: Res<RenderQueue>,
    particle_system_query: Query<(Entity, &ParticleSystem)>,
    particle_buffers_query: Query<&PreparedParticleBuffer>,
    render_pipeline: Res<ParticleRenderPipeline>,
    mut config: ResMut<ParticleConfig>,
    camera_query: Query<&ExtractedView, With<Camera>>,
    time: Res<Time>,
    mut commands: Commands,
    mut ran: Local<bool>,
)
{
    if !*ran 
    {
        *ran = true;
        info!("[Setup] Setting up particle buffers");
        if let Ok(view) = camera_query.single() {
            let view_matrix = view.world_from_view.compute_matrix().inverse();
            let view_proj = view.clip_from_view * view_matrix;
            config.view_proj = view_proj.to_cols_array_2d();
        }
        config.delta_time = time.delta().as_secs_f32();

        let uniform_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("uniform_buffer"),
            size: std::mem::size_of::<ParticleConfig>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
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
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
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
                    storage_buffer: storage_buffer,
                    uniform_buffer: uniform_buffer
                });
        }
        info!("[Setup] initialized all buffers for GPU");
    }
    else 
    {
        info!("updating particle buffers");
        // Update view_proj from camera
        if let Ok(view) = camera_query.single() {
            let view_matrix = view.world_from_view.compute_matrix().inverse();
            let view_proj = view.clip_from_view * view_matrix;
            config.view_proj = view_proj.to_cols_array_2d();
        }

        // Update time delta
        config.delta_time = time.delta().as_secs_f32();

        // Update the uniform buffer on the GPU
        if let Ok(prepared_particles) = particle_buffers_query.single() {
            render_queue.write_buffer(
                &prepared_particles.uniform_buffer,
                0,
                bytemuck::bytes_of(config.as_ref()),
            );
        }
    }
}
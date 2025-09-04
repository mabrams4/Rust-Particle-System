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
use crate::{particle_compute::SortingParams};
use crate::particle::Particle;
use crate::util::{get_bind_group_layout, get_bind_group_normal, get_bind_group_swapped, get_render_pipeline_descriptor};

#[derive(RenderLabel, Hash, Debug, Eq, PartialEq, Clone)]
pub struct ParticleRenderLabel;

#[derive(Component)]
pub struct GPUPipelineBuffers {
    pub bind_group_normal: BindGroup,  // shared between vertex and compute shaders
    pub bind_group_swapped: BindGroup,  // ping pong bind group
    pub vertex_buffer: Buffer,
    pub config_buffer: Buffer,
    pub spatial_lookup_buffer: Buffer,
    pub temp_sorting_buffer: Buffer,
    pub sorting_params_buffer: Buffer,
    //pub grid_start_idxs_buffer: Buffer,
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
        //println!("Running Render Node");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ParticleRenderPipeline>();
        let config = world.resource::<ParticleConfig>();

        for target in self.view_query.iter_manual(world) 
        {
            for entity in self.particle_system.iter_manual(world)
            {
                // check if pipeline is ready yet
                if let Some(render_pipeline_id) = pipeline_cache.get_render_pipeline(pipeline.render_pipeline_id)
                {
                    //let particle_system = world.get::<ParticleSystem>(entity).unwrap();
                    if let Some(render_pipeline_buffers) = world.get::<GPUPipelineBuffers>(entity)
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
                        render_pass.set_render_pipeline(render_pipeline_id);
                        render_pass.set_bind_group(0, &render_pipeline_buffers.bind_group_normal, &[]);
                        render_pass.set_vertex_buffer(0, render_pipeline_buffers.vertex_buffer.slice(..));
                        render_pass.draw(0..6, 0..config.particle_count as u32);
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

pub fn prepare_particle_buffers(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    particle_system_query: Query<(Entity, &ParticleSystem)>,
    pipeline_buffers_query: Query<&GPUPipelineBuffers>,
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
        if let Ok(view) = camera_query.single() {
            let view_matrix = view.world_from_view.compute_matrix().inverse();
            let view_proj = view.clip_from_view * view_matrix;
            config.view_proj = view_proj.to_cols_array_2d();
        }
        config.delta_time = time.delta().as_secs_f32();

        let config_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("uniform_buffer"),
            size: std::mem::size_of::<ParticleConfig>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let config_buffer_size = config_buffer.size();
        let config_buffer_size = std::num::NonZeroU64::new(config_buffer_size).unwrap();

        if let Ok((entity, particle_system)) = particle_system_query.single() 
        {
            let particles = &particle_system.particles;

            let mut byte_buffer = Vec::<u8>::new();
            let mut buffer = encase::StorageBuffer::new(&mut byte_buffer);
            buffer.write(&particles).unwrap();

            let particle_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {   
                label: Some("storage_buffer"), 
                contents: buffer.into_inner(), 
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let particle_buffer_size = (std::mem::size_of::<Particle>() * config.particle_count as usize) as u64;
            let particle_buffer_size = std::num::NonZeroU64::new(particle_buffer_size).unwrap();

            let spatial_lookup_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("grid_metadata_buffer"),
                size: (std::mem::size_of::<u32>() * 2 * config.particle_count as usize) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let spatial_lookup_buffer_size = spatial_lookup_buffer.size();
            let spatial_lookup_buffer_size = std::num::NonZeroU64::new(spatial_lookup_buffer_size).unwrap();

            let temp_sorting_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("temp_sorting_buffer"),
                size: (std::mem::size_of::<u32>() * 2 * config.particle_count as usize) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let sorting_params_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("sorting_config_buffer"),
                size: std::mem::size_of::<SortingParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let sorting_params_buffer_size = sorting_params_buffer.size();
            let sorting_params_buffer_size = std::num::NonZeroU64::new(sorting_params_buffer_size).unwrap();

            let grid_start_idxs_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("grid_start_idxs_buffer"),
                size: (std::mem::size_of::<u32>() * config.particle_count as usize) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let grid_start_idxs_buffer_size = grid_start_idxs_buffer.size();
            let grid_start_idxs_buffer_size = std::num::NonZeroU64::new(grid_start_idxs_buffer_size).unwrap();

            let bind_group_normal = get_bind_group_normal(
                "bind_group_normal",
                &render_device,
                &render_pipeline.bind_group_layout,
                &particle_buffer,
                particle_buffer_size,
                &config_buffer,
                config_buffer_size,
                &spatial_lookup_buffer,
                spatial_lookup_buffer_size,
                &grid_start_idxs_buffer,
                grid_start_idxs_buffer_size,
                &temp_sorting_buffer,
                &sorting_params_buffer,
                sorting_params_buffer_size,
            );
            let bind_group_swapped = get_bind_group_swapped(
                "bind_group_swapped",
                &render_device,
                &render_pipeline.bind_group_layout,
                &particle_buffer,
                particle_buffer_size,
                &config_buffer,
                config_buffer_size,
                &spatial_lookup_buffer,
                spatial_lookup_buffer_size,
                &grid_start_idxs_buffer,
                grid_start_idxs_buffer_size,
                &temp_sorting_buffer,
                &sorting_params_buffer,
                sorting_params_buffer_size,
            );

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
            
            commands.entity(entity).insert(GPUPipelineBuffers 
                {
                    bind_group_normal: bind_group_normal,
                    bind_group_swapped: bind_group_swapped,
                    vertex_buffer: vertex_buffer,
                    config_buffer: config_buffer,
                    spatial_lookup_buffer: spatial_lookup_buffer,
                    temp_sorting_buffer: temp_sorting_buffer,
                    sorting_params_buffer: sorting_params_buffer,
                    //grid_start_idxs_buffer: grid_start_idxs_buffer
                });
        }
    }
    else 
    {
        // Update view_proj from camera
        // if let Ok(view) = camera_query.single() {
        //     let view_matrix = view.world_from_view.compute_matrix().inverse();
        //     let view_proj = view.clip_from_view * view_matrix;
        //     config.view_proj = view_proj.to_cols_array_2d();
        // }
        
        // Update time delta
        config.delta_time = time.delta().as_secs_f32();
        
        // Update the uniform buffer on the GPU
        if let Ok(render_particle_buffers) = pipeline_buffers_query.single() {
            render_queue.write_buffer(
                &render_particle_buffers.config_buffer,
                0,
                bytemuck::bytes_of(config.as_ref()),
            );
        }
    }
}
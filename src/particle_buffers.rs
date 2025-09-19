use bevy::{
    prelude::*,
    render::{
        render_resource::{*}, 
        renderer::{RenderDevice, RenderQueue},
        view::ExtractedView,
    },
};

use bytemuck::{Pod, Zeroable};
use crate::ParticleSystem;
use crate::particle_render::ParticleRenderPipeline;
use crate::ParticleConfig;
use crate::particle::Particle;
use crate::util::get_bind_group;

#[derive(Component)]
pub struct GPUPipelineBuffers {
    pub bind_group: BindGroup,  // shared between vertex and compute shaders
    pub vertex_buffer: Buffer,
    pub config_buffer: Buffer,
    pub spatial_lookup_buffer: Buffer,
    pub spatial_lookup_offsets_buffer: Buffer,
    pub particle_densities_buffer: Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SortingParams    // Used for spatial lookup buffer sorting
{
    n: u32,
    group_width: u32,
    group_height: u32,
    step_index: u32
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
                size: (std::mem::size_of::<u32>() * 2 * config.particle_count.next_power_of_two() as usize) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let spatial_lookup_buffer_size = spatial_lookup_buffer.size();
            let spatial_lookup_buffer_size = std::num::NonZeroU64::new(spatial_lookup_buffer_size).unwrap();

            // BITONIC MERGE SORT STUFF
            let n = config.particle_count;
            let next_pow_2 = n.next_power_of_two();

            let num_stages = u32::ilog2(next_pow_2);
            let mut total_iterations = 0usize;
            for stage in 0..num_stages as usize {
                total_iterations += stage + 1;
            }
            const UNIFORM_ALIGNMENT: usize = 256;
            let aligned_size = ((std::mem::size_of::<SortingParams>() + UNIFORM_ALIGNMENT - 1) 
                            / UNIFORM_ALIGNMENT) * UNIFORM_ALIGNMENT;

            let sorting_params_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("Sorting Params Buffer"),
                size: (total_iterations as u64 * aligned_size as u64),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create aligned parameter data
            let mut sorting_buffer_data = vec![0u8; total_iterations * UNIFORM_ALIGNMENT];
            let mut iteration = 0;

            for stage_index in 0..num_stages {
                for step_index in 0..=stage_index {
                    let group_width = 1 << (stage_index - step_index);
                    let group_height = 2 * group_width - 1;
                    let params = SortingParams { 
                        n: next_pow_2, 
                        group_width, 
                        group_height, 
                        step_index 
                    };
                    
                    // Write at aligned offset
                    let offset = iteration * UNIFORM_ALIGNMENT;
                    sorting_buffer_data[offset..offset + std::mem::size_of::<SortingParams>()]
                        .copy_from_slice(bytemuck::bytes_of(&params));
                    
                    iteration += 1;
                }
            }

            // Write all parameters at once
            render_queue.write_buffer(&sorting_params_buffer, 0, &sorting_buffer_data);

            let spatial_lookup_offsets_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("spatial_lookup_offsets_buffer"),
                size: (std::mem::size_of::<u32>() * config.particle_count as usize) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let spatial_lookup_offsets_buffer_size = spatial_lookup_offsets_buffer.size();
            let spatial_lookup_offsets_buffer_size = std::num::NonZeroU64::new(spatial_lookup_offsets_buffer_size).unwrap();

            let particle_densities_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("particle_densities_buffer"),
                size: (std::mem::size_of::<f32>() * config.particle_count as usize) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let particle_densities_buffer_size = particle_densities_buffer.size();
            let particle_densities_buffer_size = std::num::NonZeroU64::new(particle_densities_buffer_size).unwrap();

            let bind_group = get_bind_group(
                "bind_group",
                &render_device,
                &render_pipeline.bind_group_layout,
                &particle_buffer,
                particle_buffer_size,
                &config_buffer,
                config_buffer_size,
                &spatial_lookup_buffer,
                spatial_lookup_buffer_size,
                &spatial_lookup_offsets_buffer,
                spatial_lookup_offsets_buffer_size,
                &sorting_params_buffer,
                &particle_densities_buffer,
                particle_densities_buffer_size
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
                    bind_group: bind_group,
                    vertex_buffer: vertex_buffer,
                    config_buffer: config_buffer,
                    spatial_lookup_buffer: spatial_lookup_buffer,
                    spatial_lookup_offsets_buffer: spatial_lookup_offsets_buffer,
                    particle_densities_buffer: particle_densities_buffer
                });
        }
    }
    else 
    {
        // Update time delta
        config.delta_time = time.delta().as_secs_f32();
        config.frame_count += 1;
        
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
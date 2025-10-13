use bevy::{
    prelude::*, render::{
        render_graph::{self, Node, RenderGraphContext, RenderLabel}, 
        render_resource::*, 
        renderer::{RenderContext, RenderDevice},
    }
};

use crate::{particle_compute::render_graph::NodeRunError, ParticleConfig};
use crate::ParticleSystem;
use crate::particle_buffers::GPUPipelineBuffers;
use crate::util::{get_bind_group_layout, get_compute_pipeline_descriptor};

const WORKGROUP_SIZE: u32 = 64;
const UNIFORM_ALIGNMENT: usize = 256;

#[derive(RenderLabel, Hash, Debug, Eq, PartialEq, Clone)]
pub struct ParticleComputeLabel;

#[derive(Resource)]
pub struct ParticleComputePipeline 
{
    compute_grid_pipeline_id: CachedComputePipelineId,
    compute_sort_particles_pipeline_id: CachedComputePipelineId,
    compute_spatial_lookup_offsets_pipeline_id: CachedComputePipelineId,
    compute_pre_sim_step_pipeline_id: CachedComputePipelineId,
    compute_sim_step_pipeline_id: CachedComputePipelineId,
}

impl FromWorld for ParticleComputePipeline 
{
    // called when Pipeline is created
    fn from_world(world: &mut World) -> Self 
    {
        // get render device
        let render_device = world.resource::<RenderDevice>();

        // get shader handle
        let shader_handle = world.resource::<AssetServer>().load("compute_shader.wgsl");
        
        // create the bind group layout
        let bind_group_layout = get_bind_group_layout(render_device);

        // create the render pipeline and store it in the pipeline cache
        let pipeline_cache = world.resource_mut::<PipelineCache>();
        
        // pipeline for grid creation and cell binning
        let compute_grid_pipeline_id = pipeline_cache.queue_compute_pipeline(
            get_compute_pipeline_descriptor(&bind_group_layout, &shader_handle, "bin_particles_in_grid")
        );

        // need to sort the array here
        let compute_sort_particles_pipeline_id = pipeline_cache.queue_compute_pipeline(
            get_compute_pipeline_descriptor(&bind_group_layout, &shader_handle, "sort_particles")
        );

        let compute_spatial_lookup_offsets_pipeline_id = pipeline_cache.queue_compute_pipeline(
            get_compute_pipeline_descriptor(&bind_group_layout, &shader_handle, "calculate_spatial_lookup_offsets")
        );

        let compute_pre_sim_step_pipeline_id = pipeline_cache.queue_compute_pipeline(
            get_compute_pipeline_descriptor(&bind_group_layout, &shader_handle, "pre_simulation_step")
        );

        // pipeline for main simulation step
        let compute_sim_step_pipeline_id = pipeline_cache.queue_compute_pipeline(
            get_compute_pipeline_descriptor(&bind_group_layout, &shader_handle, "simulation_step")
        );
        
        // return the ParticleComputePipeline object
        ParticleComputePipeline 
        {  
            compute_grid_pipeline_id: compute_grid_pipeline_id,
            compute_sort_particles_pipeline_id: compute_sort_particles_pipeline_id,
            compute_spatial_lookup_offsets_pipeline_id: compute_spatial_lookup_offsets_pipeline_id,
            compute_sim_step_pipeline_id: compute_sim_step_pipeline_id,
            compute_pre_sim_step_pipeline_id: compute_pre_sim_step_pipeline_id,
        }
    }
}

pub struct ParticleComputeNode 
{
    particle_system: QueryState<Entity, With<ParticleSystem>>,
}

impl Node for ParticleComputeNode 
{
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> 
    {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ParticleComputePipeline>();
        let config = world.resource::<ParticleConfig>();

        for entity in self.particle_system.iter_manual(world) {
            if let Some(pipeline_buffers) = world.get::<GPUPipelineBuffers>(entity) {

                // Pass 1: assign particles to cells in uniform grid
                {
                    let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor::default());

                    if let Some(pipeline_id_grid) = pipeline_cache.get_compute_pipeline(pipeline.compute_grid_pipeline_id)
                    {
                        pass.set_bind_group(0, &pipeline_buffers.bind_group, &[0]);
                        pass.set_pipeline(pipeline_id_grid);
                        pass.dispatch_workgroups((config.particle_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
                    }
                }
                
                // Pass 2: sort particles by grid cell key
                {
                    if let Some(pipeline_id_sort) =
                        pipeline_cache.get_compute_pipeline(pipeline.compute_sort_particles_pipeline_id)
                    {
                        let n = config.particle_count;
                        let next_pow_2 = n.next_power_of_two();

                        let num_pairs = next_pow_2 / 2;
                        let num_stages = u32::ilog2(next_pow_2);
                        let mut iteration = 0;
                        for stage_index in 0..num_stages
                        {
                            for _ in 0..=stage_index    // step_index
                            {
                                // Create pass in a scope so it's dropped after dispatch
                                {
                                    let mut pass = render_context.command_encoder()
                                        .begin_compute_pass(&ComputePassDescriptor::default());
                                    
                                    pass.set_pipeline(pipeline_id_sort);

                                    let dynamic_offset = (iteration * UNIFORM_ALIGNMENT) as u32;
                                    pass.set_bind_group(0, &pipeline_buffers.bind_group, &[dynamic_offset]);
                                    
                                    let num_workgroups = (num_pairs + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;  // 64 threads per workgroup
                                    pass.dispatch_workgroups(num_workgroups, 1, 1);
                                } // Pass is dropped here, ensuring completion
                                iteration += 1;
                            }
                        }
                    }
                }

                // Pass 3: Calculate grid start idxs
                {
                    let mut pass = render_context.command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());

                    if let Some(pipeline_id_spatial_lookup_offsets) =
                        pipeline_cache.get_compute_pipeline(pipeline.compute_spatial_lookup_offsets_pipeline_id)
                    {
                        pass.set_bind_group(0, &pipeline_buffers.bind_group, &[0]);
                        pass.set_pipeline(pipeline_id_spatial_lookup_offsets);
                        pass.dispatch_workgroups((config.particle_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
                    }
                } 

                // Pass 4: update predicted positions and particle densities
                {
                    let mut pass = render_context.command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());

                    if let Some(pipeline_id_pre_sim_step) =
                        pipeline_cache.get_compute_pipeline(pipeline.compute_pre_sim_step_pipeline_id)
                    {
                        pass.set_bind_group(0, &pipeline_buffers.bind_group, &[0]);
                        pass.set_pipeline(pipeline_id_pre_sim_step);
                        pass.dispatch_workgroups((config.particle_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
                    }
                } 

                // Pass 5: integrate particle dynamics
                {
                    let mut pass = render_context.command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());

                    if let Some(pipeline_id_sim_step) =
                        pipeline_cache.get_compute_pipeline(pipeline.compute_sim_step_pipeline_id)
                    {
                        pass.set_bind_group(0, &pipeline_buffers.bind_group, &[0]);
                        pass.set_pipeline(pipeline_id_sim_step);
                        pass.dispatch_workgroups((config.particle_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
                    }
                } 
            }
        }
        Ok(())
    }

    fn update(&mut self, world: &mut World) {
        self.particle_system.update_archetypes(world);
    }
}

impl ParticleComputeNode {
    pub fn new(world: &mut World) -> Self 
    {
        Self 
        {
            particle_system: QueryState::new(world),
        }
    }
}


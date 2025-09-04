use bevy::{
    prelude::*,
    render::{
        render_graph::{self, Node, RenderGraphContext, RenderLabel}, 
        render_resource::*, 
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};

use crate::{particle_compute::render_graph::NodeRunError, ParticleConfig};
use crate::ParticleSystem;
use crate::particle_render::GPUPipelineBuffers;
use crate::util::{get_bind_group_layout, get_compute_pipeline_descriptor};

const WORKGROUP_SIZE: u32 = 16;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SortingParams
{
    n: u32,
    j: u32,
    k: u32,
    padding: u32
}

#[derive(RenderLabel, Hash, Debug, Eq, PartialEq, Clone)]
pub struct ParticleComputeLabel;

#[derive(Resource)]
pub struct ParticleComputePipeline 
{
    compute_main_pipeline_id: CachedComputePipelineId,
    compute_grid_pipeline_id: CachedComputePipelineId,
    compute_sort_particles_pipeline_id: CachedComputePipelineId,
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

        // pipeline for main simulation step
        let compute_main_pipeline_id = pipeline_cache.queue_compute_pipeline(
            get_compute_pipeline_descriptor(&bind_group_layout, &shader_handle, "simulation_step")
        );
        
        // return the ParticleComputePipeline object
        ParticleComputePipeline 
        {  
            compute_main_pipeline_id: compute_main_pipeline_id,
            compute_grid_pipeline_id: compute_grid_pipeline_id,
            compute_sort_particles_pipeline_id: compute_sort_particles_pipeline_id,
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
        //println!("Running Compute Node");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ParticleComputePipeline>();
        let config = world.resource::<ParticleConfig>();
        let queue = world.resource::<RenderQueue>();

        for entity in self.particle_system.iter_manual(world) {
            if let Some(pipeline_buffers) = world.get::<GPUPipelineBuffers>(entity) {

                // Pass 1: assign particles to cells in uniform grid
                {
                    let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor::default());

                    if let Some(pipeline_id_grid) = pipeline_cache.get_compute_pipeline(pipeline.compute_grid_pipeline_id)
                    {
                        pass.set_bind_group(0, &pipeline_buffers.bind_group_normal, &[]);
                        pass.set_pipeline(pipeline_id_grid);
                        pass.dispatch_workgroups(config.particle_count / WORKGROUP_SIZE, 1, 1);
                    }
                }
                // Pass 2: sort particles by grid cell key
                {
                    if let Some(pipeline_id_sort) =
                        pipeline_cache.get_compute_pipeline(pipeline.compute_sort_particles_pipeline_id)
                    {
                        let n = config.particle_count;
                        let np2 = n.max(1).next_power_of_two();
                        let logn = 32 - np2.leading_zeros();

                        let mut in_buf  = &pipeline_buffers.spatial_lookup_buffer;
                        let mut out_buf = &pipeline_buffers.temp_sorting_buffer;
                        let mut use_swapped = false;

                        for stage in 0..logn 
                        {
                            for step in 0..=stage 
                            {
                                let j = 1u32 << (stage - step);
                                let k = j * 2;

                                let params = SortingParams { n, j, k, padding: 0 };
                                queue.write_buffer(
                                    &pipeline_buffers.sorting_params_buffer,
                                    0,
                                    bytemuck::bytes_of(&params),
                                );

                                // one compute pass per dispatch
                                let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor::default());
                                
                                pass.set_pipeline(pipeline_id_sort);

                                if use_swapped {
                                    pass.set_bind_group(0, &pipeline_buffers.bind_group_swapped, &[]);
                                } else {
                                    pass.set_bind_group(0, &pipeline_buffers.bind_group_normal, &[]);
                                }

                                let workgroups = (np2 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                                pass.dispatch_workgroups(workgroups, 1, 1);

                                use_swapped = !use_swapped;
                                
                                std::mem::swap(&mut in_buf, &mut out_buf);
                            }
                        }

                        if !std::ptr::eq(in_buf, &pipeline_buffers.spatial_lookup_buffer) {
                            render_context.command_encoder().copy_buffer_to_buffer(
                                in_buf,
                                0,
                                &pipeline_buffers.spatial_lookup_buffer,
                                0,
                                (np2 * std::mem::size_of::<[u32; 2]>() as u32) as u64,
                            );
                        }
                        
                        
                    }
                }

                // Pass 3: integrate particle dynamics
                {
                    let mut pass = render_context.command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());

                    if let Some(pipeline_id_main) =
                        pipeline_cache.get_compute_pipeline(pipeline.compute_main_pipeline_id)
                    {
                        pass.set_bind_group(0, &pipeline_buffers.bind_group_normal, &[]);
                        pass.set_pipeline(pipeline_id_main);
                        pass.dispatch_workgroups(config.particle_count / WORKGROUP_SIZE, 1, 1);
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


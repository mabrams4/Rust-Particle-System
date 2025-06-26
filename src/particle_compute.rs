use bevy::{
    prelude::*,
    render::{
        render_graph::{self, Node, RenderGraphContext, RenderLabel}, 
        render_resource::*, 
        renderer::{RenderContext, RenderDevice},
        view::{ExtractedView, ViewTarget, Msaa},
    },
};

use crate::{particle_compute::render_graph::NodeRunError, ParticleConfig};
use crate::ParticleSystem;
use crate::particle::Particle;
use crate::particle_render::PreparedParticleBuffer;

const WORKGROUP_SIZE: u32 = 16;

#[derive(RenderLabel, Hash, Debug, Eq, PartialEq, Clone)]
pub struct ParticleComputeLabel;

#[derive(Resource)]
pub struct ParticleComputePipeline 
{
    bind_group_layout: BindGroupLayout,
    compute_pipeline_id: CachedComputePipelineId,
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

        info!("[S-Compute] created bind group layout");

        // create the render pipeline and store it in the pipeline cache
        let pipeline_cache = world.resource_mut::<PipelineCache>();

        let compute_pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor 
            {   label: Some("compute_pipeline_id".into()), 
                layout: vec![bind_group_layout.clone()],
                push_constant_ranges: vec![], 
                shader: shader_handle.clone(), 
                shader_defs: vec![], 
                entry_point: "compute_main".into(), 
                zero_initialize_workgroup_memory: false 
            });
        
        info!("[S] created compute pipeline");
        // return the ParticleRenderPipeline object
        ParticleComputePipeline 
        {  
            bind_group_layout,
            compute_pipeline_id
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
        for entity in self.particle_system.iter_manual(world)
        {
            // check if pipeline is ready yet
            if let Some(pipeline_id) = pipeline_cache.get_compute_pipeline(pipeline.compute_pipeline_id)
            {
                //let particle_system = world.get::<ParticleSystem>(entity).unwrap();
                let prepared_particle_buffer = world.get::<PreparedParticleBuffer>(entity).unwrap();
                let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor::default());
                
                pass.set_bind_group(0, &prepared_particle_buffer.bind_group, &[]);
                pass.set_pipeline(pipeline_id);
                pass.dispatch_workgroups(prepared_particle_buffer.num_particles / WORKGROUP_SIZE, 1, 1);

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
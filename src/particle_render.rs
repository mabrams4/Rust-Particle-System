use bevy::{
    prelude::*,
    render::{
        render_graph::{self, Node, RenderGraphContext, RenderLabel}, 
        render_resource::{*}, 
        renderer::{RenderContext, RenderDevice},
        view::ViewTarget,
    },
};

use crate::{particle_render::render_graph::NodeRunError, ParticleConfig};
use crate::ParticleSystem;
use crate::particle_buffers::GPUPipelineBuffers;
use crate::util::{get_bind_group_layout, get_render_pipeline_descriptor};


#[derive(RenderLabel, Hash, Debug, Eq, PartialEq, Clone)]
pub struct ParticleRenderLabel;

#[derive(Resource)]
pub struct ParticleRenderPipeline 
{
    pub bind_group_layout: BindGroupLayout, // shared with compute shader
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
        
        // get bind group layout
        let bind_group_layout = get_bind_group_layout(render_device);

        // create the render pipeline and store it in the pipeline cache
        let pipeline_cache = world.resource_mut::<PipelineCache>();

        // queue the render pipeline
        let render_pipeline_id = pipeline_cache.queue_render_pipeline(
            get_render_pipeline_descriptor(&bind_group_layout, &shader_handle)
        );

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
                    // check if pipeline buffers are ready
                    if let Some(render_pipeline_buffers) = world.get::<GPUPipelineBuffers>(entity)
                    {
                        // create render pass and set attributes
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
                        render_pass.set_bind_group(0, &render_pipeline_buffers.bind_group, &[0]);
                        render_pass.set_vertex_buffer(0, render_pipeline_buffers.vertex_buffer.slice(..));
                        render_pass.draw(0..6, 0..config.particle_count as u32);
                    }
                }
            }
        }
        Ok(())
    }

    // update ECS state
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


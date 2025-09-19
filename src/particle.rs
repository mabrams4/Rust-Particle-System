use bevy::{
    prelude::*,
    render::{
        Render,
        extract_component::ExtractComponentPlugin, 
        extract_resource::ExtractResourcePlugin, 
        graph::CameraDriverLabel, 
        render_graph::RenderGraph, 
        render_resource::*, 
        RenderApp, RenderSet,
    },
};

use crate::{ParticleConfig, ParticleSystem};
use crate::particle_render::{ParticleRenderNode, ParticleRenderLabel, ParticleRenderPipeline};
use crate::particle_buffers::prepare_particle_buffers;
use crate::particle_compute::{ParticleComputeNode, ParticleComputeLabel, ParticleComputePipeline};
use crate::debug::{ParticleDebugLabel, ParticleDebugNode};

#[derive(ShaderType, Default, Clone, Copy)] 
pub struct Particle {
    pub position: [f32; 2],
    pub velocity: [f32; 2], 
    pub color: [f32; 4],
}

pub struct ParticlePlugin;

impl Plugin for ParticlePlugin 
{
    fn build(&self, app: &mut App) 
    {
        // extract particle system to render world
        app.add_plugins(ExtractComponentPlugin::<ParticleSystem>::default());
        app.add_plugins(ExtractResourcePlugin::<ParticleConfig>::default());

        // get render app

        let render_app = app.sub_app_mut(RenderApp);
        
        render_app.add_systems(Render, prepare_particle_buffers.in_set(RenderSet::Prepare));

        // Create the render node
        let render_node = ParticleRenderNode::new(render_app.world_mut());
        let compute_node = ParticleComputeNode::new(render_app.world_mut());
        let debug_node = ParticleDebugNode::new(render_app.world_mut());

        // get the render graph
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();

        // add the node and node edge to the render graph
        render_graph.add_node(ParticleRenderLabel, render_node);
        render_graph.add_node(ParticleComputeLabel, compute_node);
        render_graph.add_node(ParticleDebugLabel, debug_node);

        render_graph.add_node_edge(ParticleComputeLabel, ParticleDebugLabel);
        render_graph.add_node_edge(ParticleDebugLabel, ParticleRenderLabel);
        render_graph.add_node_edge(ParticleRenderLabel, CameraDriverLabel);

    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        // insert Custom Particle Pipelines into render world
        render_app.init_resource::<ParticleComputePipeline>();
        render_app.init_resource::<ParticleRenderPipeline>();
    }
}

use bevy::{
    prelude::*,
    render::{
        render_graph::{self, Node, RenderGraphContext, RenderLabel}, 
        render_resource::*, 
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};

use crate::{debug::render_graph::NodeRunError, ParticleConfig};
use crate::ParticleSystem;
use crate::particle_render::GPUPipelineBuffers;


#[derive(RenderLabel, Hash, Debug, Eq, PartialEq, Clone)]
pub struct ParticleDebugLabel;

pub struct ParticleDebugNode
{
    particle_system: QueryState<Entity, With<ParticleSystem>>,
}

impl Node for ParticleDebugNode 
{
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        _render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> 
    {
        //println!("START DEBUG NODE");
        let particle_count = world.resource::<ParticleConfig>().particle_count;
        let queue = world.resource::<RenderQueue>();
        let device = world.resource::<RenderDevice>();

        for entity in self.particle_system.iter_manual(world) {
            if let Some(pipeline_buffers) = world.get::<GPUPipelineBuffers>(entity) 
            {
                let result = read_buffer_from_gpu(
                    device, 
                    queue, 
                    &pipeline_buffers.spatial_lookup_buffer, 
                    particle_count
                );
                validate_spatial_lookup(result, particle_count);
            }
        }
        //println!("END DEBUG NODE");
        Ok(())
    }

    fn update(&mut self, world: &mut World) {
        self.particle_system.update_archetypes(world);
    }
}

impl ParticleDebugNode {
    pub fn new(world: &mut World) -> Self 
    {
        Self 
        {
            particle_system: QueryState::new(world),
        }
    }
}

pub fn read_buffer_from_gpu(
    device: &RenderDevice,
    queue: &RenderQueue, 
    source_buffer: &Buffer,
    particle_count: u32
) -> Vec<[u32; 2]> {
    //println!("Reading Spatial Lookup Buffer");
    let buffer_size = (particle_count as u64) * std::mem::size_of::<[u32; 2]>() as u64;
    
    // Create staging buffer
    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy operation
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source_buffer, 0, &staging_buffer, 0, buffer_size);
    queue.submit(std::iter::once(encoder.finish()));

    // Synchronous mapping using std::sync::mpsc
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    
    buffer_slice.map_async(MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    
    device.poll(Maintain::wait()).panic_on_timeout();
    
    receiver.recv().unwrap().unwrap();

    // Read data
    let data = buffer_slice.get_mapped_range();
    
    let result: Vec<[u32; 2]> = bytemuck::cast_slice(&data).to_vec();
    
    // Cleanup
    drop(data);
    staging_buffer.unmap();
    
    result
}

pub fn validate_spatial_lookup(
    array: Vec<[u32; 2]>,
    particle_count: u32,
) {
    for i in 1..particle_count as usize {
        if array[i][0] < array[i - 1][0] {
            println!("Array is NOT sorted at index {i}");
            let end = (i + 10).min(particle_count as usize);
            for j in i..end {
                println!("Index {j}: Cell Key {}, Particle Index {}", array[j][0], array[j][1]);
            }
            return;
        }
    }
    println!("ARRAY IS SORTED!!!");
}
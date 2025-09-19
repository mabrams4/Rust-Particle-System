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
use crate::particle_buffers::GPUPipelineBuffers;

const DEBUG: bool = false;


#[derive(RenderLabel, Hash, Debug, Eq, PartialEq, Clone)]
pub struct ParticleDebugLabel;

pub struct ParticleDebugNode
{
    particle_system: QueryState<Entity, With<ParticleSystem>>,
    frame_count: u32,
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
        if DEBUG
        {
            //println!("START DEBUG NODE");
            let particle_count = world.resource::<ParticleConfig>().particle_count;
            let queue = world.resource::<RenderQueue>();
            let device = world.resource::<RenderDevice>();

            for entity in self.particle_system.iter_manual(world) {
                if let Some(pipeline_buffers) = world.get::<GPUPipelineBuffers>(entity) 
                {
                    // let spatial_lookup = read_spatial_lookup_buffer_from_gpu(
                    //     device, 
                    //     queue, 
                    //     &pipeline_buffers.spatial_lookup_buffer, 
                    //     particle_count
                    // );
                    // validate_spatial_lookup(spatial_lookup, particle_count);

                    // let spatial_lookup_offsets = read_grid_start_idxs_from_gpu(
                    //     device, 
                    //     queue, 
                    //     &pipeline_buffers.spatial_lookup_offsets_buffer, 
                    //     particle_count
                    // );
                    // print_spatial_lookup_offsets(spatial_lookup_offsets, particle_count);
                    let densities = read_particle_densities_from_gpu(
                        device, 
                        queue, 
                        &pipeline_buffers.particle_densities_buffer, 
                        particle_count
                    );
                    if self.frame_count % 10 == 0
                    {
                        print_densities(densities, self.frame_count);
                    }
                }
            }
            //println!("END DEBUG NODE");
            }
        Ok(())
    }

    fn update(&mut self, world: &mut World) {
        self.particle_system.update_archetypes(world);
        self.frame_count += 1;
    }
}

impl ParticleDebugNode {
    pub fn new(world: &mut World) -> Self 
    {
        Self 
        {
            particle_system: QueryState::new(world),
            frame_count: 0,
        }
    }
}

pub fn read_spatial_lookup_buffer_from_gpu(
    device: &RenderDevice,
    queue: &RenderQueue, 
    source_buffer: &Buffer,
    particle_count: u32
) -> Vec<[u32; 2]> {
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
    println!("SPATIAL LOOKUP");
    for i in 0..particle_count as usize {
        println!("Index {}: Cell Key {}", i, array[i][0]);
    }
    //println!("ARRAY IS SORTED!!!");
}

pub fn read_grid_start_idxs_from_gpu(
    device: &RenderDevice,
    queue: &RenderQueue, 
    source_buffer: &Buffer,
    particle_count: u32
) -> Vec<u32> {
    let buffer_size = (particle_count as u64) * std::mem::size_of::<u32>() as u64;
    
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
    
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    
    // Cleanup
    drop(data);
    staging_buffer.unmap();
    
    result
}

pub fn read_particle_densities_from_gpu(
    device: &RenderDevice,
    queue: &RenderQueue, 
    source_buffer: &Buffer,
    particle_count: u32
) -> Vec<f32> {
    let buffer_size = (particle_count as u64) * std::mem::size_of::<f32>() as u64;
    
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
    
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    
    // Cleanup
    drop(data);
    staging_buffer.unmap();
    
    result
}

pub fn print_densities(
    densities: Vec<f32>,
    frame_count: u32,
)
{
    println!("PARTICLE DENSITIES");
    for i in 0..10 as usize {
        println!("Frame {frame_count}: Index {}: Value {}", i, densities[i]);
    }
}

pub fn print_spatial_lookup_offsets(
    array: Vec<u32>,
    particle_count: u32,
) {
    println!("SPATIAL LOOKUP OFFSETS");
    for i in 0..particle_count as usize {
        println!("Index {}: Value {}", i, array[i]);
    }
    //println!("ARRAY IS SORTED!!!");
}
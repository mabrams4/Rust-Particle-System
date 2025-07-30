use bevy::{
    prelude::*,
    //asset::RenderAssetUsages,
    render::{
        extract_component::ExtractComponent, 
        extract_resource::ExtractResource, 
        //mesh::{Indices, PrimitiveTopology},
    },
};
use bytemuck::{Pod, Zeroable};

// use bevy_inspector_egui::{
//     bevy_egui::EguiPlugin,
//     quick::WorldInspectorPlugin,
// };

mod particle;
mod particle_render;
mod particle_compute;
mod util;
use particle::Particle;



#[derive(ExtractComponent, Component, Default, Clone)]
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
    //pub mesh_handle: Handle<Mesh>,
}

#[repr(C)]
#[derive(ExtractResource, Resource, Default, Clone, Copy, Zeroable, Pod)]
pub struct ParticleConfig {
    pub particle_count: u32,
    pub particle_size: f32,
    pub delta_time: f32,
    pub gravity: f32,
    pub view_proj: [[f32; 4]; 4],
}

const PARTICLE_COUNT: u32 = 1000;
const PARTICLE_SIZE: f32 = 5.0;
const PARTICLE_SPACING: f32 = 7.5;
const GRAVITY: f32 = 100.0;

fn main() 
{
    App::new()
    .add_plugins(DefaultPlugins)
    //.add_plugins(EguiPlugin { enable_multipass_for_primary_context: true })
    //.add_plugins(WorldInspectorPlugin::new())

    .add_plugins(particle::ParticlePlugin)

    .insert_resource(ParticleConfig {
        particle_count: PARTICLE_COUNT,
        particle_size: PARTICLE_SIZE,
        view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        delta_time: 0.0,
        gravity: GRAVITY
    })

    .add_systems(Startup, setup)
    .add_systems(Update, exit_on_escape)
    .run();
}

fn setup(mut commands: Commands, particle_config: Res<ParticleConfig>)
{
    let total_particles = particle_config.particle_count;
    let grid_width = (total_particles as f32).sqrt().ceil() as u32;
    let grid_height = (total_particles as f32 / grid_width as f32).ceil() as u32;

    let spacing = PARTICLE_SPACING; // adjust as needed
    let half_width = (grid_width as f32 - 1.0) * spacing / 2.0;
    let half_height = (grid_height as f32 - 1.0) * spacing / 2.0;
    let mut particles = Vec::with_capacity(total_particles as usize);
    for row in 0..grid_height {
        for col in 0..grid_width {
            if particles.len() >= total_particles as usize {
                break;
            }

            let x = col as f32 * spacing - half_width;
            let y = row as f32 * spacing - half_height;
            particles.push(Particle {
                position: [x, y],
                velocity: [0.0, 0.0],
                acceleration: [0.0, -GRAVITY],
                temp1: 0.0,
                temp2: 0.0,
                color: [1.0, 1.0, 1.0, 1.0],
            });
        }
    }
    commands.spawn(
        ParticleSystem {
            particles: particles,
            //mesh_handle: mesh_handle,
        },
    );
    commands.spawn(Camera2d::default());
    info!("[Setup] Spawned ParticleSystem in Main World");
}

fn exit_on_escape(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut exit: EventWriter<AppExit>,
) {
    if keyboard_input.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}
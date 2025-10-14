use bevy::{
    prelude::*,
    render::{
        extract_component::ExtractComponent, 
        extract_resource::ExtractResource, 
    },
    window::WindowMode,
};
use rand_distr::{Distribution, Normal};
use bytemuck::{Pod, Zeroable};
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};

use std::f32::consts::PI;

mod particle;
mod particle_render;
mod particle_compute;
mod util;
mod debug;
mod particle_buffers;
mod parameter_gui;
use particle::Particle;
use parameter_gui::{gui_system, apply_gui_updates, GUIConfig};

const PARTICLE_COUNT: u32 = 50000;
const PARTICLE_SIZE: f32 = 3.0;
const SMOOTHING_RADIUS: f32 = PARTICLE_SIZE * PARTICLE_SIZE;
const GRAVITY: f32 = 0.0;
const TARGET_DENSITY: f32 = 0.011;
const PRESSURE_MULTIPLIER: f32 = 10000.0;
const NEAR_DENSITY_MULTIPLIER: f32 = 1000.0;
const VISCOCITY_STRENGTH: f32 = 5.0;
const DAMPING_FACTOR: f32 = 0.1;
const FIXED_DELTA_TIME: f32 = 1.0 / 100.0;
const MAX_ENERGY: f32 = 2000.0;

#[derive(ExtractComponent, Component, Default, Clone)]
pub struct ParticleSystem 
{
    pub particles: Vec<Particle>,
}

#[repr(C)]
#[derive(ExtractResource, Resource, Default, Clone, Copy, Zeroable, Pod)]
pub struct ParticleConfig {
    pub particle_count: u32,            // 4 bytes
    pub particle_size: f32,             // 4 bytes
    pub smoothing_radius: f32,          // 4 bytes
    pub max_energy: f32,                // 4 bytes

    pub damping_factor: f32,            // 4 bytes
    pub fixed_delta_time: f32,          // 4 bytes
    pub frame_count: u32,               // 4 bytes
    pub gravity: f32,                   // 4 bytes

    pub density_kernel_norm: f32,       // 4 bytes
    pub near_density_kernel_norm: f32,  // 4 bytes
    pub viscocity_kernel_norm: f32,     // 4 bytes
    pub _padding: f32,                  // 4 bytes

    pub target_density: f32,            // 4 bytes
    pub pressure_multiplier: f32,       // 4 bytes
    pub viscocity_strength: f32,        // 4 bytes
    pub near_density_multiplier: f32,   // 4 bytes

    pub screen_bounds: [f32; 4],        // 16 bytes     [x_min, x_max, y_min, y_max]

    pub view_proj: [[f32; 4]; 4],       // 64 bytes
}

fn main() 
{
    App::new()
    .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                mode: WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
                ..default()
            }),
            ..default()
        }))
    .add_plugins(particle::ParticlePlugin)
    .add_plugins(EguiPlugin::default())

    // Actual simulation parameters used in compute shader
    .insert_resource(ParticleConfig {
        particle_count: PARTICLE_COUNT,
        particle_size: PARTICLE_SIZE,
        smoothing_radius: SMOOTHING_RADIUS,  // Also our grid cell size
        max_energy: MAX_ENERGY,

        damping_factor: DAMPING_FACTOR,
        fixed_delta_time: FIXED_DELTA_TIME,
        frame_count: 0,
        gravity: GRAVITY,

        density_kernel_norm: 10.0 / (PI * SMOOTHING_RADIUS.powf(5.0)),
        near_density_kernel_norm: 15.0 / (PI * SMOOTHING_RADIUS.powf(6.0)),
        viscocity_kernel_norm: 4.0 / (PI * SMOOTHING_RADIUS.powf(8.0)),
        _padding: 0.0,

        target_density: TARGET_DENSITY,
        pressure_multiplier: PRESSURE_MULTIPLIER,
        viscocity_strength: VISCOCITY_STRENGTH,
        near_density_multiplier: NEAR_DENSITY_MULTIPLIER,

        screen_bounds: [0.0; 4],
        view_proj: Mat4::IDENTITY.to_cols_array_2d(),
    })
    
    // GUI modifiable sim params
    .insert_resource(GUIConfig {
        fixed_delta_time: FIXED_DELTA_TIME,
        smoothing_radius: SMOOTHING_RADIUS,
        max_energy: MAX_ENERGY,

        gravity: GRAVITY,
        damping_factor: DAMPING_FACTOR,
        target_density: TARGET_DENSITY,
        pressure_multiplier: PRESSURE_MULTIPLIER,
        
        viscocity_strength: VISCOCITY_STRENGTH,
        near_density_multiplier: NEAR_DENSITY_MULTIPLIER,
        applied_changes: false,  
    })  

    

    .add_systems(Startup, setup_camera)
    .add_systems(PreUpdate, apply_gui_updates)
    .add_systems(EguiPrimaryContextPass, gui_system)
    .add_systems(Update, setup_particles)
    .add_systems(Update, exit_on_escape)
    .run();
}

fn get_screen_bounds(
    camera_query: &Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) -> Option<[f32; 4]> 
{
    let (camera, transform) = camera_query.single().ok()?;
    let viewport_size = camera.logical_viewport_size()?;

    let center = transform.translation().truncate();
    let half_width = viewport_size.x / 2.0;
    let half_height = viewport_size.y / 2.0;

    Some([
        center.x - half_width, // x_min
        center.x + half_width, // x_max
        center.y - half_height, // y_min
        center.y + half_height, // y_max
    ])
}

fn setup_camera(mut commands : Commands)
{
    commands.spawn(Camera2d::default());
}

fn setup_particles(
    commands: Commands,
    mut particle_config: ResMut<ParticleConfig>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    mut ran: Local<bool>
) {
    if !*ran
    {
        *ran = true;

        // Get and store screen bounds
        if let Some(bounds) = get_screen_bounds(&camera_query) {
            particle_config.screen_bounds = bounds;
        } else {
            warn!("[Setup] Failed to retrieve screen bounds from camera query");
            return; // Exit setup early if bounds are unavailable
        }

        setup_particles_scatter(particle_config, commands);
    }
}

fn setup_particles_scatter(
    particle_config: ResMut<ParticleConfig>,
    mut commands: Commands,
)
{
    let [x_min, x_max, y_min, y_max] = particle_config.screen_bounds;
    let mut rng = rand::rng();

    // Y-distribution: mean at center
    let y_center = (y_min + y_max) / 2.0;
    let y_std_dev = (y_max - y_min) * 0.125;
    let y_dist = Normal::new(y_center, y_std_dev).unwrap();

    let mut particles = Vec::with_capacity(PARTICLE_COUNT as usize);

    // for i in 0..total_particles {
    for i in 0..PARTICLE_COUNT {
        // Uniformly distribute x across visible width
        let t = i as f32 / PARTICLE_COUNT as f32;
        let x = x_min + t * (x_max - x_min);

        // Sample y and clamp to bounds
        let mut y = y_dist.sample(&mut rng);
        y = y.clamp(y_min, y_max);

        particles.push(Particle {
            position: [x, y],
            velocity: [0.0, 0.0], 
            color: [1.0, 1.0, 1.0, 1.0],
        });
    }

    // Spawn particle system and camera
    commands.spawn(ParticleSystem { particles });
}

fn exit_on_escape(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut exit: EventWriter<AppExit>,
) {
    if keyboard_input.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

use bevy::{
    prelude::*,
    render::{
        extract_component::ExtractComponent, 
        extract_resource::ExtractResource, 
    },
    window::WindowMode,
};
use rand::prelude::*;
//use rand_distr::{Distribution, Normal};
use bytemuck::{Pod, Zeroable};

mod particle;
mod particle_render;
mod particle_compute;
mod util;
mod debug;
use particle::Particle;

const PARTICLE_COUNT: u32 = 100000;
const PARTICLE_SIZE: f32 = 3.0;
const GRAVITY: f32 = 0.0;
const COMPUTE_SHADER_DELAY: u32 = 3;
const INFLOW_VEL: f32 = 0.0;
const VERTICAL_JITTER: f32 = 0.1;
const AIR_DENSITY: f32 = 0.0;
const AIR_VISCOSITY: f32 = 0.0;
const MAX_ENERGY: f32 = 20000.0;
const PRESSURE_GRADIENT: [f32; 2] = [0.0, 0.0];

//const RANDOM_INIT_X_ACCEL: f32 = 150.0;

// TEST for exploding out of nowhere
const MAX_INIT_VEL: f32 = 200.0;
const MAX_INIT_ACCEL: f32 = 50.0;

#[derive(ExtractComponent, Component, Default, Clone)]
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
}

#[repr(C)]
#[derive(ExtractResource, Resource, Default, Clone, Copy, Zeroable, Pod)]
pub struct ParticleConfig {
    pub particle_count: u32,            // 4 bytes
    pub particle_size: f32,             // 4 bytes
    pub delta_time: f32,                // 4 bytes
    pub gravity: f32,                   // 4 bytes

    pub inflow_vel: f32,                // 4 bytes
    pub vertical_jitter: f32,           // 4 bytes
    pub air_density: f32,               // 4 bytes
    pub air_viscosity: f32,             // 4 bytes
    pub pressure_gradient: [f32; 2],    // 8 bytes

    pub padding: [f32; 2],              // 8 bytes

    pub screen_bounds: [f32; 4],        // 16 bytes     [x_min, x_max, y_min, y_max]
    pub view_proj: [[f32; 4]; 4],       // 64 bytes
    pub max_energy: f32,
    pub smoothing_radius: f32,
    pub temp3: f32,
    pub temp4: f32,
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
    //.add_plugins(DefaultPlugins)
    .add_plugins(particle::ParticlePlugin)

    .insert_resource(ParticleConfig {
        particle_count: PARTICLE_COUNT,
        particle_size: PARTICLE_SIZE,
        delta_time: 0.0,
        gravity: GRAVITY,

        inflow_vel: INFLOW_VEL,
        vertical_jitter: VERTICAL_JITTER,
        air_density: AIR_DENSITY,
        air_viscosity: AIR_VISCOSITY,
        pressure_gradient: PRESSURE_GRADIENT,
        padding: [0.0, 0.0],

        screen_bounds: [0.0; 4],
        view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        max_energy: MAX_ENERGY,
        smoothing_radius: 5.0 * PARTICLE_SIZE,  // Also our grid cell size
        temp3: 0.0,
        temp4: 0.0,
    })

    .add_systems(Startup, setup_camera)
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
    mut commands: Commands,
    mut particle_config: ResMut<ParticleConfig>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    mut ran: Local<bool>
) {
    if !*ran
    {
        *ran = true;

        let total_particles = particle_config.particle_count;

        // Get and store screen bounds
        if let Some(bounds) = get_screen_bounds(&camera_query) {
            particle_config.screen_bounds = bounds;
        } else {
            warn!("[Setup] Failed to retrieve screen bounds from camera query");
            return; // Exit setup early if bounds are unavailable
        }
        // let [x_min, x_max, y_min, y_max] = particle_config.screen_bounds;
        // let mut rng = rand::rng();

        // // Y-distribution: mean at center
        // let y_center = (y_min + y_max) / 2.0;
        // let y_std_dev = (y_max - y_min) * 0.125;
        // let y_dist = Normal::new(y_center, y_std_dev).unwrap();

        let mut particles = Vec::with_capacity(total_particles as usize);

        // for i in 0..total_particles {
        for _ in 0..total_particles {
            // Uniformly distribute x across visible width
            // let t = i as f32 / total_particles as f32;
            // let x = x_min + t * (x_max - x_min);
            // // Sample y and clamp to bounds
            // let mut y = y_dist.sample(&mut rng);
            // y = y.clamp(y_min, y_max);

            // // Small vertical velocity jitter
            // let y_velocity = rng.random_range(-VERTICAL_JITTER..VERTICAL_JITTER);
            // let x_accel = rng.random_range(-RANDOM_INIT_X_ACCEL..RANDOM_INIT_X_ACCEL);

            // particles.push(Particle {
            //     position: [x, y],
            //     velocity: [INFLOW_VEL, y_velocity], // rightward + small variation
            //     acceleration: [x_accel, particle_config.gravity],
            //     compute_shader_delay: COMPUTE_SHADER_DELAY,
            //     temp2: 0.0,
            //     color: [1.0, 1.0, 1.0, 1.0],
            // });

            // TEST for exploding out of nowwhere
            let mut rng = rand::rng();

            let init_x_vel = rng.random_range(-MAX_INIT_VEL..MAX_INIT_VEL);
            let init_y_vel = rng.random_range(-MAX_INIT_VEL..MAX_INIT_VEL);
            let init_x_accel = rng.random_range(-MAX_INIT_ACCEL..MAX_INIT_ACCEL);
            let init_y_accel = rng.random_range(-MAX_INIT_ACCEL..MAX_INIT_ACCEL);

            particles.push(Particle {
                position: [0.0, 0.0],
                velocity: [init_x_vel, init_y_vel], // rightward + small variation
                acceleration: [init_x_accel, init_y_accel],
                compute_shader_delay: COMPUTE_SHADER_DELAY,
                temp2: 0.0,
                color: [1.0, 1.0, 1.0, 1.0],
            });
        }

        // Spawn particle system and camera
        commands.spawn(ParticleSystem { particles });
    }
}

fn exit_on_escape(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut exit: EventWriter<AppExit>,
) {
    if keyboard_input.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}
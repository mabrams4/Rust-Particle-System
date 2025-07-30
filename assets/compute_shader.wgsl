struct Config {
    particle_count: u32,    // 4 bytes
    particle_size: f32,     // 4 bytes
    delta_time: f32,        // 4 bytes
    gravity: f32,           // 4 bytes
    view_proj: mat4x4<f32>, // 64 bytes
}

struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    acceleration: vec2<f32>,
    temp1: f32,
    temp2: f32,
    color: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> config: Config;

@compute @workgroup_size(16, 1, 1)
fn compute_main()
{
    
}
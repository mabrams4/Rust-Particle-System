struct Config {
    particle_count: u32,            // 4 bytes
    particle_size: f32,             // 4 bytes
    smoothing_radius: f32,          // 4 bytes
    max_energy: f32,                // 4 bytes

    damping_factor: f32,            // 4 bytes
    pad1: f32,                      // 4 bytes
    pad2: f32,                      // 4 bytes
    pad3: f32,                      // 4 bytes

    delta_time: f32,                // 4 bytes
    fixed_delta_time: f32,          // 4 bytes
    frame_count: u32,               // 4 bytes
    gravity: f32,                   // 4 bytes

    target_density: f32,            // 4 bytes
    pressure_multiplier: f32,       // 4 bytes
    viscocity_strength: f32,        // 4 bytes
    near_density_multiplier: f32,   // 4 bytes

    screen_bounds: vec4<f32>,       // 16 bytes     [x_min, x_max, y_min, y_max]
    view_proj: mat4x4<f32>,         // 64 bytes
}

struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: vec4<f32>,
}

struct VertexInput {
    @builtin(instance_index) instance_id: u32,
    @location(0) quad_pos: vec2<f32>, // Local quad vertex position (-1 to 1)
    @location(1) uv: vec2<f32>, 
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> config: Config;

// =============================================================================
// VERTEX SHADER
// =============================================================================

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Get the particle from the storage buffer
    let particle = particles[input.instance_id];

    // Calculate quad vertex offset scaled by particle size
    let local_offset = input.quad_pos * config.particle_size;

    // World-space position of this vertex
    let world_position = vec2<f32>(particle.position + local_offset);

    // Convert to homogeneous vec4 (z = 0.0, w = 1.0)
    let world_position_4d = vec4<f32>(world_position, 0.0, 1.0);

    // Transform to clip space using view-projection matrix from uniform buffer
    output.position = config.view_proj * world_position_4d;

    output.uv = input.uv;
    output.color = particle.color;

    return output;
}

// =============================================================================
// FRAGMENT SHADER  
// =============================================================================

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> 
{
    let centered_uv = input.uv - vec2(0.5);
    let dist = length(centered_uv);

    let radius = 0.5;
    let edge_thickness = 0.1; // adjust softness

    // Compute smooth alpha
    let alpha = 1.0 - smoothstep(radius - edge_thickness, radius, dist);

    // Optional: discard very transparent pixels (optimization)
    if (alpha < 0.01) {
        discard;
    }

    return input.color; // orange-ish glowing circle
}
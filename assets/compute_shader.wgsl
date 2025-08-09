struct Config {
    particle_count: u32,            // 4 bytes
    particle_size: f32,             // 4 bytes
    delta_time: f32,                // 4 bytes
    gravity: f32,                   // 4 bytes

    inflow_vel: f32,                // 4 bytes
    vertical_jitter: f32,           // 4 bytes
    air_density: f32,               // 4 bytes
    air_viscosity: f32,             // 4 bytes

    pressure_gradient: vec2<f32>,   // 8 bytes
    padding: vec2<f32>,             // 8 bytes (ensures 16-byte alignment)

    screen_bounds: vec4<f32>,       // 16 bytes
    view_proj: mat4x4<f32>,         // 64 bytes
    max_energy: f32,
    smoothing_radius: f32,
    grid_cell_size: u32,
    temp4: f32,
};

struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    acceleration: vec2<f32>,
    compute_shader_delay: u32,
    temp2: f32,
    color: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> config: Config;

@group(0) @binding(2) 
var<storage, read_write> spatial_lookup: array<vec2<u32>>;

@group(0) @binding(3) 
var<storage, read_write> grid_start_idxs: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn simulation_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    // Early exit if delay is still active
    if (particles[i].compute_shader_delay > 0u) {
        particles[i].compute_shader_delay -= 1u;
        return;
    }

    // Recycle particle to left if off-screen
    // if (particles[i].position.x > config.screen_bounds[1]) {
    //     particles[i].position.x = config.screen_bounds[0];
    // }

    // Screen Bounds check
    let damping_Factor = 0.7;
    let x_min = config.screen_bounds[0];
    let x_max = config.screen_bounds[1];
    let y_min = config.screen_bounds[2];
    let y_max = config.screen_bounds[3];
    if (particles[i].position.x > x_max || particles[i].position.x < x_min)
    {
        particles[i].velocity.x *= -damping_Factor;
    }
    if (particles[i].position.y > y_max || particles[i].position.y < y_min)
    {
        particles[i].velocity.y *= -damping_Factor;
    }

    // Update particle position and velocity
    particles[i].position += particles[i].velocity * config.delta_time;
    particles[i].velocity += particles[i].acceleration * config.delta_time;

    // === Color Based on Kinetic Energy ===

    let mass = 1.0;                    // placeholder constant mass
    let speed_sq = dot(particles[i].velocity, particles[i].velocity);
    let energy = 0.5 * mass * speed_sq;

    let normalized = clamp(energy / config.max_energy, 0.0, 1.0);

    var rgb: vec3<f32>;
    if (normalized < 0.5) {
        let t = normalized * 2.0;
        rgb = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), t); // Blue → Green
    } else {
        let t = (normalized - 0.5) * 2.0;
        rgb = mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), t); // Green → Red
    }

    particles[i].color = vec4<f32>(rgb, 1.0);
}

fn hash_cell(cell_x: u32, cell_y: u32) -> u32
{
    let a = cell_x * 15823u; 
    let b = cell_y * 9737333u; 
    return a + b;
}

fn get_key_from_hash(hash_value: u32) -> u32
{
    return hash_value % config.particle_count;
}

@compute @workgroup_size(64, 1, 1)
fn bin_particles_in_grid(@builtin(global_invocation_id) id: vec3<u32>)
{
    let i = id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    let x_max = config.screen_bounds[1];
    let y_max = config.screen_bounds[3];

    let cell_x = u32((particles[i].position.x + x_max) / f32(config.grid_cell_size));
    let cell_y = u32((particles[i].position.y + y_max) / f32(config.grid_cell_size));

    let cell_key = get_key_from_hash(hash_cell(cell_x, cell_y));
    spatial_lookup[i] = vec2(cell_key, i);
    grid_start_idxs[i] = 0u; // placeholder 
}


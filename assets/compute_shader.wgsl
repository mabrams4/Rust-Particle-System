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
    padding2: vec4<f32>,
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

@compute @workgroup_size(64, 1, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    // Early exit if delay is still active
    if (particles[i].compute_shader_delay > 0) {
        particles[i].compute_shader_delay -= 1;
        return;
    }

    // Update particle position and velocity
    particles[i].position += particles[i].velocity * config.delta_time;
    particles[i].velocity += particles[i].acceleration * config.delta_time;

    // Recycle particle to left if off-screen
    if (particles[i].position.x > config.screen_bounds[1]) {
        particles[i].position.x = config.screen_bounds[0];
    }

    // === Color Based on Kinetic Energy ===

    let mass = 1.0;                    // placeholder constant mass
    let speed_sq = dot(particles[i].velocity, particles[i].velocity);
    let energy = 0.5 * mass * speed_sq;

    let max_energy = 10000.0;
    let normalized = clamp(energy / max_energy, 0.0, 1.0);

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

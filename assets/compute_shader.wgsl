/* ----------------------------------- STRUCTS -----------------------------------*/
struct Config {
    particle_count: u32,            // 4 bytes
    particle_size: f32,             // 4 bytes
    delta_time: f32,                // 4 bytes
    gravity: f32,                   // 4 bytes

    target_density: f32,            // 4 bytes
    pressure_multiplier: f32,       // 4 bytes
    max_energy: f32,                // 4 bytes
    smoothing_radius: f32,          // 4 bytes

    screen_bounds: vec4<f32>,       // 16 bytes
    view_proj: mat4x4<f32>,         // 64 bytes

    frame_count: u32,               // 4 bytes
    temp2: u32,                     // 4 bytes
    temp3: u32,                     // 4 bytes
    temp4: u32,                     // 4 bytes
}

struct SortingParams
{
    n: u32,
    group_width: u32,
    group_height: u32,
    step_index: u32
}

struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: vec4<f32>,
}

/* ----------------------------------- BINDINGS -----------------------------------*/
@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> config: Config;

@group(0) @binding(2)
var<uniform> sorting_params: SortingParams;

@group(0) @binding(3) 
var<storage, read_write> spatial_lookup: array<vec2<u32>>;

@group(0) @binding(4) 
var<storage, read_write> spatial_lookup_offsets: array<u32>;

@group(0) @binding(5) 
var<storage, read_write> particle_densities: array<f32>;

/* --------------------------------- CONSTANTS ---------------------------------*/
const PI: f32 = 3.14159;
const SHADER_DELAY: u32 = 5u;

/* --------------------------------- UTIL FUNCTIONS ---------------------------------*/
fn particle_position_to_cell_coord(i: u32) -> vec2<i32>
{
    let x_max = config.screen_bounds[1];
    let y_max = config.screen_bounds[3];

    let cell_x = i32((particles[i].position.x + x_max) / f32(config.smoothing_radius));
    let cell_y = i32((particles[i].position.y + y_max) / f32(config.smoothing_radius));

    return vec2(cell_x, cell_y);
}

fn hash_cell(cell_x: i32, cell_y: i32) -> u32
{
    let a = u32(cell_x) * 15823u; 
    let b = u32(cell_y) * 9737333u; 
    return a + b;
}

fn get_key_from_hash(hash_value: u32) -> u32
{
    return hash_value % config.particle_count;
}

fn smoothing_kernel(distance: f32) -> f32
{
    let radius = config.smoothing_radius;
    if (distance >= radius) { return 0f; }

    let volume = PI * pow(radius, 4f) / 6f;
    return (radius - distance) * (radius - distance) / volume;
}

fn smoothing_kernel_derivative(distance: f32) -> f32
{
    let radius = config.smoothing_radius;
    if (distance >= radius) { return 0f; }

    let scale = 12f / (pow(radius, 4f) * PI);
    return (distance - radius) * scale;
}

fn density_to_pressure_force(density: f32) -> f32
{
    return (density - config.target_density) * config.pressure_multiplier;
}

fn update_particle_density(particle_index: u32)
{
    let density = calculate_density(particle_index);
    particle_densities[particle_index] = density;
}

var<private> GRID_OFFSETS: array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(-1,-1), vec2<i32>(-1,0), vec2<i32>(-1,1),
    vec2<i32>( 0,-1), vec2<i32>( 0,0), vec2<i32>( 0,1),
    vec2<i32>( 1,-1), vec2<i32>( 1,0), vec2<i32>( 1,1)
);

fn calculate_density(curr_particle_index: u32) -> f32
{
    var density = 0f;

    let x_max = config.screen_bounds[1];
    let y_max = config.screen_bounds[3];

    let curr_particle = particles[curr_particle_index];
    let curr_particle_position = curr_particle.position;

    let cell_x = i32((curr_particle_position.x + x_max) / f32(config.smoothing_radius));
    let cell_y = i32((curr_particle_position.y + y_max) / f32(config.smoothing_radius));

    let sqr_radius = config.smoothing_radius * config.smoothing_radius;

    for (var i: u32; i < 9u; i++)
    {
        let offset = GRID_OFFSETS[i];
        let neighbor_cell_x = cell_x + offset.x;
        let neighbor_cell_y = cell_y + offset.y;

        let curr_cell_key = get_key_from_hash(hash_cell(neighbor_cell_x, neighbor_cell_y));
        let start_idx = spatial_lookup_offsets[curr_cell_key];   // calculate start idx of this cell key within spatial lookup

        // loop through neighboring particles
        for (var i: u32 = start_idx; i < config.particle_count; i++)
        {
            // break when we reach a new cell key
            let other_particle_cell_key = spatial_lookup[i][0];
            if (other_particle_cell_key != curr_cell_key) { break; }

            // skip if comparing particle against itself
            let other_particle_index = spatial_lookup[i][1];
            //if (other_particle_index == curr_particle_index) { continue; }

            let other_particle = particles[other_particle_index];

            let delta = curr_particle_position - other_particle.position;
            let sqr_distance = length(delta);

            // skip if particle not within squared radius
            if (sqr_distance > sqr_radius) { continue; }

            let distance = dot(delta, delta);
            density += smoothing_kernel(distance);
        }
    }
    return density;
}

fn calculate_shared_pressure(d1: f32, d2: f32) -> f32
{
    let p1 = density_to_pressure_force(d1);
    let p2 = density_to_pressure_force(d2);
    return (p1 + p2) / 2f;
}

fn calculate_pressure_force(curr_particle_index: u32) -> vec2<f32>
{
    var pressure = vec2(0f, 0f);

    let x_max = config.screen_bounds[1];
    let y_max = config.screen_bounds[3];

    let curr_particle = particles[curr_particle_index];
    let curr_particle_position = curr_particle.position;

    let cell_x = i32((curr_particle_position.x + x_max) / f32(config.smoothing_radius));
    let cell_y = i32((curr_particle_position.y + y_max) / f32(config.smoothing_radius));

    let sqr_radius = config.smoothing_radius * config.smoothing_radius;

    for (var i: u32; i < 9u; i++)
    {
        let offset = GRID_OFFSETS[i];
        let neighbor_cell_x = cell_x + offset.x;
        let neighbor_cell_y = cell_y + offset.y;

        let curr_cell_key = get_key_from_hash(hash_cell(neighbor_cell_x, neighbor_cell_y));
        let start_idx = spatial_lookup_offsets[curr_cell_key];   // calculate start idx of this cell key within spatial lookup

        // loop through neighboring particles
        for (var i: u32 = start_idx; i < config.particle_count; i++)
        {
            // break when we reach a new cell key
            let other_particle_cell_key = spatial_lookup[i][0];
            if (other_particle_cell_key != curr_cell_key) { break; }

            // skip if comparing particle against itself
            let other_particle_index = spatial_lookup[i][1];
            if (other_particle_index == curr_particle_index) { continue; }

            let other_particle = particles[other_particle_index];

            let delta = curr_particle_position - other_particle.position;
            let sqr_distance = dot(delta, delta);

            // skip if particle not within sqr radius
            if (sqr_distance > sqr_radius) { continue; }

            var direction = sign(delta);
            let distance = length(delta);
            if (distance == 0f) { direction = vec2(0f, 1f); }

            let slope = smoothing_kernel_derivative(distance);
            let density = particle_densities[curr_particle_index];
            let shared_pressure = calculate_shared_pressure(density, particle_densities[other_particle_index]);
            pressure += -direction * shared_pressure * slope / density;
        }
    }
    return pressure;
}


/* ----------------------------------- FUNCTIONS -----------------------------------*/
fn check_screen_bounds(i: u32) 
{
    let damping_Factor = 0.5;
    let x_min = config.screen_bounds[0];
    let x_max = config.screen_bounds[1];
    let y_min = config.screen_bounds[2];
    let y_max = config.screen_bounds[3];

    if (particles[i].position.x > x_max || particles[i].position.x < x_min) {
        particles[i].velocity.x *= -damping_Factor;
    }
    if (particles[i].position.y > y_max || particles[i].position.y < y_min) {
        particles[i].velocity.y *= -damping_Factor;
    }
}

fn set_color(i: u32) 
{
    let mass = 1.0; // constant mass
    let speed_sq = dot(particles[i].velocity, particles[i].velocity);
    let energy = 0.5 * mass * speed_sq;

    let normalized = clamp(energy / config.max_energy, 0.0, 1.0);

    var rgb: vec3<f32>;
    if (normalized < 0.5) {
        let t = normalized * 2.0;
        rgb = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), t); // Blue → Green
    } else {
        let t = (normalized - 0.5) * 2.0;
        rgb = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t); // Green → Red
    }

    particles[i].color = vec4(rgb, 1.0);
}

fn update_particle_positions(i: u32)
{
    particles[i].position += particles[i].velocity * config.delta_time;
}

fn update_particle_velocities(i: u32)
{
    // Gravity
    particles[i].velocity += vec2(0.0, -config.gravity) * config.delta_time;

    // Pressure
    let pressure_force = calculate_pressure_force(i);
    particles[i].velocity += pressure_force / particle_densities[i] * config.delta_time;
}

/* ----------------------------------- ENTRY POINT FUNCTIONS -----------------------------------*/
@compute @workgroup_size(64, 1, 1)
fn simulation_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    // Early exit if delay is still active
    // if (particles[i].compute_shader_delay > 0u) {
    //     particles[i].compute_shader_delay -= 1u;
    //     return;
    // }

    if (config.frame_count < SHADER_DELAY) { return; }

    // // Recycle particle to left if off-screen
    // if (particles[i].position.x > config.screen_bounds[1]) {
    //     particles[i].position.x = config.screen_bounds[0];
    // }

    update_particle_density(i);

    update_particle_velocities(i);

    update_particle_positions(i);

    check_screen_bounds(i);
    
    set_color(i);
}

@compute @workgroup_size(64, 1, 1)
fn bin_particles_in_grid(@builtin(global_invocation_id) id: vec3<u32>)
{
    let i = id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    let cell = particle_position_to_cell_coord(i);
    let cell_key = get_key_from_hash(hash_cell(cell.x, cell.y));
    
    spatial_lookup[i] = vec2(cell_key, i);
    spatial_lookup_offsets[i] = 0xFFFFFFFFu; // placeholder 
}

@compute @workgroup_size(64, 1, 1)
fn sort_particles(@builtin(global_invocation_id) id: vec3<u32>) 
{
    let i = id.x;
    if (i >= sorting_params.n / 2u) { return; }

    let group_width = sorting_params.group_width;
    let group_height = sorting_params.group_height;
    let step_index = sorting_params.step_index;

    let h_index = i & (group_width - 1u);
    let index_left = h_index + (group_height + 1u) * (i / group_width);
    var right_step_size: u32;  
    if (step_index == 0u)
    {
        right_step_size = group_height - 2u * h_index;
    }
    else 
    {
        right_step_size = (group_height + 1u) / 2u;
    }
    let index_right = index_left + right_step_size;
    
    // Exit if out of bounds (for non-power of 2 input sizes)
	if (index_right >= sorting_params.n) { return; }

    let value_left = spatial_lookup[index_left][0];
    let value_right = spatial_lookup[index_right][0];

    if (value_left > value_right)
    {
        let temp = spatial_lookup[index_left];
        spatial_lookup[index_left] = spatial_lookup[index_right];
        spatial_lookup[index_right] = temp;
    }
}

@compute @workgroup_size(64, 1, 1)
fn calculate_spatial_lookup_offsets(@builtin(global_invocation_id) id: vec3<u32>)
{
    let i = id.x;
    if (i >= config.particle_count) { return; }

    let key = spatial_lookup[i][0];
    var key_prev = 0xFFFFFFFFu;
    
    if (i > 0u)
    {
        key_prev = spatial_lookup[i - 1u][0];
    }

    if (key != key_prev)
    {
        spatial_lookup_offsets[key] = i;
    }
}



